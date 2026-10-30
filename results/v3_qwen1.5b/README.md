# v3 scores

The agent harness changed in v3. Search lists tables and logs, python reads a bundled file by path, a count printed by python or equal to the matching lines in the cited text stays grounded, and the critic must return one JSON object.

`single_base`, `team_base`, `team_lora`, and `team_distill` were each scored once on the frozen test under this harness. `comparison.json` holds the paired bootstrap intervals between them. The repository README renders both from these files with `python -m orbit_agents.eval.readme`.

The v2 numbers in `results/v2_qwen1.5b/` stay archived. Compare the four runs here with each other, not with the v2 table.

The notebook cell gated by `ORBIT_RESCORE=1` wrote these files. It does not train, it skips a file that is already measured, and it does not invent a score when an adapter is missing.
