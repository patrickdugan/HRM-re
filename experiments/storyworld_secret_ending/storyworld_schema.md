# Storyworld JSON Schema (Minimal)

This experiment uses a minimal, self-contained storyworld schema (not SweepWeave). The environment expects a JSON file with the following structure:

```
{
  "id": "...",
  "variables": {
    "var_name": {"min": 0, "max": 100, "start": 50}
  },
  "hidden_variables": {
    "flag_name": {"type": "bool", "start": false}
  },
  "nodes": [
    {
      "id": "node_id",
      "text": "Narrative text shown to the agent.",
      "choices": [
        {
          "id": "choice_id",
          "text": "Choice text shown to the agent.",
          "to": "next_node_id",
          "effects": {"visible_var": 5},
          "hidden_effects": {"flag_name": true}
        }
      ]
    }
  ],
  "endings": [
    {
      "id": "ending_id",
      "type": "normal|secret",
      "node_id": "node_id",
      "condition": "rationality>=90 AND favor>=80 AND flag_name==true"
    }
  ]
}
```

Notes:
- `effects` and `hidden_effects` are optional. Missing fields default to no change.
- Conditions are simple boolean expressions using `AND` / `OR`, comparison operators, visible variables, and hidden flags.
- Endings are evaluated on each step after transitioning into the next node.
- If multiple endings match, the first match in the list is used.

## SweepWeave / Storyworld Reader Compatibility
If the JSON includes `encounters` and `spools`, the environment will mirror `storyworld_reader.html` behavior:
- Uses the first reaction for each option.
- Applies only `Bounded Number Effect` nudges.
- Visible variables are the player character's non-`p*` properties.
- Endings are any encounter in the spool whose name/id includes \"ending\".
- A hidden `secret_path` flag is set when choosing any option whose id contains `_secret_`.
