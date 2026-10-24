# Orbit Code split

Protocol id: `code-eval-v1`.

The frozen files are `tasks.jsonl` and everything under `repos/`. `manifest.json` stores the bundle sha256 and the split tag. The scorer checks that hash. When the git tag `code-eval-v1` is present, it also checks the files at that tag.

Hidden tests live in each repo's `hidden/` directory. The oracle lives in `oracle/`. Neither is copied into the agent workspace, and neither is placed in the prompt. The visible tests are copied so pytest can run, and the reviewer rejects a diff that touches them.

Do not edit these files to chase a score. Add a new protocol instead.
