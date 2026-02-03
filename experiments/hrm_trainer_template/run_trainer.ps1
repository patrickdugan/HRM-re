param(
  [string]$ConfigPath = "trainer_config.json"
)

if (-not (Test-Path $ConfigPath)) {
  Write-Error "Config not found: $ConfigPath"
  exit 1
}

$config = Get-Content $ConfigPath | ConvertFrom-Json

$maxRamMb = [int]$config.max_ram_mb
$maxCpuPct = [int]$config.max_cpu_pct

if ($maxRamMb -le 0 -or $maxCpuPct -le 0) {
  Write-Error "Invalid caps in config."
  exit 1
}

$memoryLimitBytes = $maxRamMb * 1MB
$cpuRatePermille = [int]($maxCpuPct * 100)

Add-Type -TypeDefinition @"
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

public static class JobObject {
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
  public static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

  [DllImport("kernel32.dll")]
  public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

  [DllImport("kernel32.dll")]
  public static extern bool SetInformationJobObject(IntPtr hJob, int infoType, IntPtr lpJobObjectInfo, uint cbJobObjectInfoLength);

  public const int JobObjectExtendedLimitInformation = 9;
  public const int JobObjectCpuRateControlInformation = 15;
  public const uint JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100;
  public const uint JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x1;
  public const uint JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x4;

  [StructLayout(LayoutKind.Sequential)]
  public struct IO_COUNTERS {
    public ulong ReadOperationCount;
    public ulong WriteOperationCount;
    public ulong OtherOperationCount;
    public ulong ReadTransferCount;
    public ulong WriteTransferCount;
    public ulong OtherTransferCount;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
    public long PerProcessUserTimeLimit;
    public long PerJobUserTimeLimit;
    public uint LimitFlags;
    public UIntPtr MinimumWorkingSetSize;
    public UIntPtr MaximumWorkingSetSize;
    public uint ActiveProcessLimit;
    public long Affinity;
    public uint PriorityClass;
    public uint SchedulingClass;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
    public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
    public IO_COUNTERS IoInfo;
    public UIntPtr ProcessMemoryLimit;
    public UIntPtr JobMemoryLimit;
    public UIntPtr PeakProcessMemoryUsed;
    public UIntPtr PeakJobMemoryUsed;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct JOBOBJECT_CPU_RATE_CONTROL_INFORMATION {
    public uint ControlFlags;
    public uint CpuRate;
  }
}
"@

$job = [JobObject]::CreateJobObject([IntPtr]::Zero, "hrm-trainer")
if ($job -eq [IntPtr]::Zero) {
  Write-Error "Failed to create Job Object."
  exit 1
}

# Memory cap
$limit = New-Object JobObject+JOBOBJECT_EXTENDED_LIMIT_INFORMATION
$limit.BasicLimitInformation.LimitFlags = [JobObject]::JOB_OBJECT_LIMIT_PROCESS_MEMORY
$limit.ProcessMemoryLimit = [UIntPtr]$memoryLimitBytes
$size = [System.Runtime.InteropServices.Marshal]::SizeOf($limit)
$ptr = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
[System.Runtime.InteropServices.Marshal]::StructureToPtr($limit, $ptr, $false)
[JobObject]::SetInformationJobObject($job, [JobObject]::JobObjectExtendedLimitInformation, $ptr, $size) | Out-Null
[System.Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)

# CPU cap (hard cap)
$cpu = New-Object JobObject+JOBOBJECT_CPU_RATE_CONTROL_INFORMATION
$cpu.ControlFlags = [JobObject]::JOB_OBJECT_CPU_RATE_CONTROL_ENABLE -bor [JobObject]::JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP
$cpu.CpuRate = $cpuRatePermille
$size2 = [System.Runtime.InteropServices.Marshal]::SizeOf($cpu)
$ptr2 = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size2)
[System.Runtime.InteropServices.Marshal]::StructureToPtr($cpu, $ptr2, $false)
[JobObject]::SetInformationJobObject($job, [JobObject]::JobObjectCpuRateControlInformation, $ptr2, $size2) | Out-Null
[System.Runtime.InteropServices.Marshal]::FreeHGlobal($ptr2)

Write-Host "Starting trainer with caps: ${maxRamMb}MB RAM, ${maxCpuPct}% CPU"

$proc = Start-Process -FilePath "python" -ArgumentList "train_stub.py --config $ConfigPath" -PassThru -NoNewWindow
[JobObject]::AssignProcessToJobObject($job, $proc.Handle) | Out-Null
$proc.WaitForExit()
exit $proc.ExitCode
