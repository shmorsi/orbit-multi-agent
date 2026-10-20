# Team eval protocol

Protocol id: `team-eval-v1`.

The frozen file is `eval/test.jsonl`. Its sha256 is recorded in `eval/manifest.json`. Tag `team-eval-v1` points at the commit that introduced that file. Later commits may change code. They may not change the test file.

`eval/dev.jsonl` is the only source for training dialogues. The first 16 dev tasks are the train subset. The last 8 are tune. Six tasks are traps that the corpus cannot answer. Gold answers were produced by running the tools, then checked a second way inside the sandbox.

A scripted run is `kind: system_check`. It proves the harness. It is not a language-model score.

A model score is `kind: model_eval` and `status: measured`, written by `python -m orbit_agents.eval.run` with a transformers, Ollama, or llama.cpp backend. `python -m orbit_agents.eval.compare` builds bootstrap intervals only from three measured model files. Pending files stay pending.

Do not edit `eval/test.jsonl` to chase a score.
