# Orbit Code split v2

Protocol id: `code-eval-v2`.

The frozen files are `tasks.jsonl` and everything under `repos/`. `manifest.json` stores the bundle sha256 and the split tag. The scorer checks that hash. When the git tag `code-eval-v2` is present, it also checks the files at that tag.

This split is separate from `code-eval-v1`. Do not edit `data/scenarios/code/` to change v2, and do not edit these files to chase a score.

Hidden tests live in each repo's `hidden/` directory. The oracle lives in `oracle/`. A `partial/` directory, when present, is a careless patch that can pass the visible test and fail the hidden one. None of those directories is copied into the agent workspace, and none is placed in the prompt. The visible tests are copied so pytest can run, and the reviewer rejects a diff that touches them.

Some tasks pass the visible test on the buggy source and fail the hidden tests. Some need more than one source file changed before the visible test passes. `last-n` tells the coder that the visible test is the spec. A diff that edits that test is still a reviewer rejection.

`python data/scenarios/code_v2/build.py` rewrites this split. Do not run it to refresh a checkout that is already frozen.
