# Pi — First Session Guide for This Project

This file assumes Pi is already installed.

## 1. Start Pi in the repository root

Your current project path from recent terminal output is:

```bash
cd ~/Documents/153/cyberwm_starter
```

Confirm:

```bash
pwd
ls
```

You should see files/directories such as:

```text
README.md
requirements.txt
scripts/
src/
configs/
lab/
```

After adding this context pack you should also see:

```text
AGENTS.md
MEMORY.md
TODO.md
docs/
```

Then launch:

```bash
pi
```

Pi works relative to the directory where you launch it, so **always start it from the repository root for this project**.

---

## 2. Authenticate

Inside Pi:

```text
/login
```

Choose the provider/account you want to use.

Pi's current official quickstart supports subscription login options and API-key providers.

---

## 3. Pick a model / thinking level

Inside Pi:

```text
/model
```

Choose the model you want for coding/reasoning.

Use:

```text
/thinking
```

to set reasoning depth.

For architecture, debugging, and research-heavy changes, use a stronger reasoning setting.
For simple file edits, a cheaper/faster setting may be enough.

---

## 4. How project context works

Pi core automatically loads project `AGENTS.md` files at startup.

This pack deliberately puts the main invariant instructions in root:

```text
AGENTS.md
```

`AGENTS.md` tells Pi to read:

```text
MEMORY.md
docs/PROJECT_CONTEXT.md
docs/CURRENT_STATE.md
docs/DATA_PIPELINE.md
docs/MITRE_MAPPING.md
docs/LEAKAGE_AND_SPLITS.md
docs/ROADMAP.md
docs/RESEARCH_CONTEXT.md
TODO.md
```

If you edit context files while Pi is already running, restart Pi or run:

```text
/reload
```

### Important note about `MEMORY.md`

Core Pi does **not** automatically treat a project-root `MEMORY.md` as built-in long-term memory.
We keep it because:

1. `AGENTS.md` explicitly tells Pi to read it;
2. it is versionable with the project;
3. it provides a durable handoff between sessions/agents.

An optional third-party `pi-memory` package exists, but you do not need it for this workflow.

---

## 5. Before letting Pi modify anything: checkpoint with git

Pi has write/edit/bash tools and can modify files in the current repository.

First check whether git is already initialized:

```bash
git status
```

If it is a repository, make a checkpoint:

```bash
git add -A
git commit -m "checkpoint before pi work"
```

If it is not yet a git repository and you want version control:

```bash
git init
git add -A
git commit -m "initial cyber world model checkpoint"
```

Do not commit giant raw datasets/PCAPs unless you intentionally configure storage for them.
Add raw/generated heavy artifacts to `.gitignore` as appropriate.

---

## 6. The first prompt to give Pi

Paste this:

```text
Read AGENTS.md, MEMORY.md, TODO.md, and all markdown files under docs/ that AGENTS.md requires. Then inspect the actual repository and the current lab_001 outputs. Do not modify anything yet.

I want you to:
1. tell me what is actually implemented versus only planned;
2. verify the current temporal/state invariants from the files;
3. inspect why the current state timeline is sparse;
4. inspect why the second srv1->srv2 T1021.004 hop does not appear in state_ground_truth.csv;
5. propose the smallest correct patch that creates dense fixed 5-second states and robust short-event alignment without introducing leakage.

For every finding, distinguish observed fact from inference. Keep the final goal as a predictive world model, not an IDS classifier.
```

This forces Pi to inspect rather than guess.

---

## 7. Referencing files

Pi lets you reference files with `@`.

Examples:

```text
@lab/episodes/lab_001/ground_truth.csv explain exactly how these events are aligned
```

```text
@scripts/03_build_graph_states.py review this for temporal leakage and sparse-window bugs
```

```text
@docs/DATA_PIPELINE.md update this after you verify the implementation
```

You can type `@` in the editor to fuzzy-search files.

---

## 8. Running shell commands from Pi

In interactive Pi:

```text
!command
```

runs a shell command and sends the output into model context.

Example:

```text
!head -30 lab/episodes/lab_001/state_ground_truth.csv
```

Use:

```text
!!command
```

when you want to run a command without adding its output to model context.

---

## 9. Useful session commands

Pi saves sessions automatically.

Continue the most recent session:

```bash
pi -c
```

Browse earlier sessions:

```bash
pi -r
```

Inside a session:

```text
/session
```

shows the session file/ID.

```text
/tree
```

navigates branches.

```text
/fork
```

creates a new branch from an earlier point.

```text
/compact
```

summarizes older messages when context grows large.

Use a fresh branch/session for major alternative architecture experiments rather than mixing incompatible decisions into one long thread.

---

## 10. Recommended working rhythm

For each meaningful task, ask Pi to follow this pattern:

```text
Inspect -> explain -> propose -> patch -> run -> inspect output -> document
```

Do not accept:

```text
patch -> "looks good"
```

without output verification.

A good request looks like:

```text
Inspect the current graph-state builder and one real output file. Explain the invariant we need, patch only what is necessary, run the lab pipeline, inspect the output timestamps and ATT&CK alignment, and update CURRENT_STATE.md/TODO.md with verified facts.
```

---

## 11. Keep generated data visible to Pi

Pi can only inspect files it can access in the local filesystem.

Keep project outputs under the repository, e.g.:

```text
lab/episodes/
outputs/
```

or tell Pi their exact external path.

When a command generates an output that matters, ask Pi to inspect the file directly instead of pasting only a summary.

---

## 12. Optional long-term Pi memory package

Pi's official package directory currently lists a third-party package called `pi-memory`.

If you later want user-wide automatic memory across sessions:

```bash
pi install npm:pi-memory
```

It stores memory under Pi's agent directory, not the repository.

However:

- this is optional;
- third-party Pi packages can execute code and affect agent behavior;
- review the package/source before installing it.

For this research repo, the checked-in `AGENTS.md` + `MEMORY.md` + `docs/` workflow is already enough and is easier to audit.

---

## 13. Do not let Pi do these without explicit reason

- train an IDS classifier and call it the world model;
- delete/rewrite raw data;
- create random train/test row splits;
- encode ground-truth labels into observations;
- assume an ATT&CK mapping without evidence;
- generate network attacks outside the private lab;
- add large architectural complexity before the data invariants are correct.

---

## 14. Where to ask "what next?"

Use:

```text
Read TODO.md and CURRENT_STATE.md, inspect the files required for the first unchecked NOW item, and execute only that milestone. Explain each step and update the docs after verification.
```

That keeps the project moving without losing the final goal.
