# Prompt: how to write up an investigation in this folder

Use this when turning a body of work into reports here. The goal is that someone six months from now
can find whether a thing was tried, what it cost, and what it meant — without reading prose or
re-running anything.

## Structure

```
YYYY-MM_topic/
  README.md      index, down to individual test level
  00_setup.md    shared setup + scope limits + standing caveats, once
  01_....md      one file per QUESTION, several tests inside
  02_....md
```

One folder per investigation. Never put a study's setup at the top level — a later study with a
different schedule or baseline will silently inherit it.

File names are the question, in lowercase words: `03_can_the_model_be_forced_to_use_it.md`.

## Every test gets its own block

Give each test a stable ID and never renumber. Prefixes: `P` = probe on existing data, no compute.
`M` = measurement/analysis. `T` = a run that cost compute.

```markdown
## T8 — Blank the worker's own columns during training
`run15_longf_od30` (0.3), `run16_longf_od60` (0.6)

**Idea.** One line: why this was worth trying.
**How.** The mechanism, and the non-obvious choices. Most of your budget goes here.
**Result.** The number. No prose.
**Verdict.** One line: what it means.
```

If two tests share an idea but differ in a setting, they are still two blocks, or one block with both
runs named — never a bullet buried in a paragraph.

## Where to spend the writing budget

**Spend it on How.** The numbers are in the records; the reasoning is not. Write the things that
cannot be recovered from a CSV:

- why the test is built the way it is, especially where a simpler version would be wrong
- the control, and what it rules out
- what you verified before trusting it
- the choice you had to make and the one you rejected

Example of the difference. Bad: *"trained with dropout on the own-worker columns."* Good: *"blanked
on training rows only — inference never blanks. Signalled by a learned `nn.Embedding(2,4)` rather
than a raw bit. Own columns located at `[client_id*window, +window)` so nothing is reordered."*

**Do not spend it on Result or Verdict.** One line each. A result is a number with its units and its
comparison point. A verdict is a claim, not a hedge — if it is uncertain, say what would settle it.

**Do not spend it on narrative.** No "we then decided to", no chronology. The order tests were run in
is not information.

**Negative results get the same budget as positive ones.** They are the main reason this folder
exists. So do traps — a measurement that looked real and was not is worth a block, because otherwise
someone repeats it.

## The index must stand alone

`README.md` carries a table per file, one row per test: ID, one-line name, verdict. Someone must be
able to read every conclusion of the study without opening a single part file. If a verdict does not
fit in a table cell, the verdict is not sharp enough yet.

End with a short **Verdict** section: what the study concluded, and what is still open.

## Rules that keep it retrievable

- Every claim names the run directory or script that produced it.
- Every number carries its comparison point and its noise floor. A bare number is not a result.
- State the scope: what this tested, and what it therefore does *not* settle. A negative result about
  one implementation is not a negative result about the idea.
- Caveats live in `00_setup.md`, once. Do not repeat them per file.
- Say it once. If the same sentence is in two files, one of them should point at the other.
- When a later result overturns an earlier one, edit the earlier claim and say it was overturned.
  Do not leave both standing.
