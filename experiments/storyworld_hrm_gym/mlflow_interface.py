import importlib


def maybe_init_mlflow(cfg):
    if not cfg or not cfg.get("enabled", False):
        return None
    try:
        mlflow = importlib.import_module("mlflow")
    except Exception:
        return None

    uri = cfg.get("tracking_uri")
    if uri:
        mlflow.set_tracking_uri(uri)
    exp = cfg.get("experiment")
    if exp:
        mlflow.set_experiment(exp)
    return mlflow


def log_params(mlflow, params):
    if not mlflow:
        return
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(mlflow, metrics):
    if not mlflow:
        return
    for k, v in metrics.items():
        mlflow.log_metric(k, v)


def log_artifact(mlflow, path):
    if not mlflow or not path:
        return
    mlflow.log_artifact(path)
