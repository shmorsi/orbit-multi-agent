# v3 scores

The agent harness changed in v3. Search lists tables and logs, python reads a bundled file by path, a count printed by python or equal to the matching lines in the cited text stays grounded, and the critic must return one JSON object.

The v2 numbers in `results/v2_qwen1.5b/` stay archived. `single_base`, `team_base`, `team_lora`, and `team_distill` have not been scored again on the frozen test under this harness. Do not compare them with the v2 table until that rescore exists.

The notebook cell gated by `ORBIT_RESCORE=1` writes those four files here, plus `comparison.json`. It does not train, and it does not invent a score when an adapter is missing.
