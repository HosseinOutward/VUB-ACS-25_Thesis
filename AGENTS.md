# Project Context

This repository supports PhD-level research into gradient compression for federated learning using distributed source coding, especially learned Wyner-Ziv quantization (LWZQ) methods.

The core research direction is a two-step compression pipeline:
1. A distributed source coding quantizer, preferably a learned Wyner-Ziv quantizer.
2. A conditional entropy coder, ideally based on Slepian-Wolf coding. 

notes on conditional entropy coder: 
    - this is not the topic of our research, but is part of the pipeline.
    - the prior rate of the current learned wz quantizer is an estimate of the rate of the SW coding.

The central design question is always: what side information is available at the decoder, and can it legitimately be used as a prior?


---

## 0. Research Principles

- **rate-distortion** - our goal is to reduce RD, not just rate or distortion in isolation. Each LWZQ stage is a point on a method's RD curve. Curves are compared by the mean rate difference at equal distortion over their shared distortion range, reading each curve piecewise-linearly (`Experiments/Protocol_test/rd_metrics.py`). the current optimal curve is [(), (), ()] (wmape db, prior bit)
- **Two-step compression.** Keep quantization and entropy coding conceptually separate. The quantizer produces symbols; the conditional entropy coder compresses those symbols using decoder-available side information.
- **Decoder-side priors matter.** Both the DSC quantizer and the conditional entropy coder need priors. Before using a prior, identify where decoding happens and which data is actually available there. Do not assume worker-only or server-only information is available everywhere.
- **Multi-stage learned quantization per float.** A learned quantizer emit several vectors of indices per float. Each stage can use earlier stage outputs as side information for later stages, producing multiple low-bit symbols from one floating-point value.
- **Structural validation only.** Pydantic, validation, formats, enums, and required fields. though also don't bomb a code based with a manual check for everthing. This is mostly to validate assumptions where it wont automaticall cause any error or wanrning.

---

## 1. Coding Standards

Note: FYI. the .venv is a symbolink, and has all the packages.

These standards apply to every module unless a more specific local instruction says otherwise.

- **Explicit caching only.** Production code currently performs every expensive operation directly. If caching is introduced for a specific operation, declare its directory, complete semantic match value, and computation together at that call site with `CacheEntry`; keep reusable match builders beside the semantics they describe. Never add generic cache fields to shared configuration or pass cache policy through unrelated layers.
- **Use full type hints** on every function signature and class attribute.
- **Use advanced python syntax**, Its encouraged to use python best practices and advanced syntax to improve the code logic and readability. avoid over engineering for the sake of syntax though. for example, `pathlib.Path` for filesystem paths over raw `str`, and many other uses of tools given by python and common libraries.
- **Keep code to a minimum** Common thread among below standards too, is to make use of dynamic approaches, judgment of if an extra code had any real impact or use, and if an extra function and class needed to be made. Less is always better mostly since it helps with readibilty and debugging. all else is secondary.
- **Write concise docstrings.** Public functions need a one-paragraph docstring. Each shared data contract needs a class-level docstring explaining what it represents and who produces and consumes it.
- **Comments explain why, not what.** Comment only when intent is non-obvious, such as a conditional field requirement or artifact ordering. Do not narrate self-evident code.
- **Fail loudly and clearly.** Hidden behavior is worse than a visible failure. On error, raise with a message that names the offending file, field, or stage. Avoid silent fallbacks, (backward) compatibility logic, and unnecessary recovery logic.
- **No bare `except`.** Catch specific exceptions only when handling them is useful. Use assertions for internal assumptions.
- **rasising errors** Instead of if-raises, use asserts where its a valid replacement. also only use the asserts where either an assumption has to be checked, or some future error will happen which might not be clear. for the latter, this means avoid asserts just to make a custom text for fail, when its already obvious. instead use it where downstream fails are not clear or worse, not even caught.
- **No global mutable state.** Pass `Config` and other dependencies explicitly. Do not introduce module-level mutable singletons.
- **Avoid needless abstraction.** Do not add tiny one-use functions or renaming-only helpers. Add abstractions only when they remove real duplication or make a complex concept easier to reason about.
- **Keep control flow shallow.** Avoid unnecessary nested loops and conditionals. Prefer direct, readable logic.
- **Avoid meaningless intermediate variables.** Create new names only when they clarify a non-trivial transformation or prevent repeated complex expressions.
- **Format by logical structure, not a rigid line length.** Keep short imports, calls, signatures, and type annotations on one line; when wrapping is genuinely helpful, group related arguments or expressions instead of scattering every small item onto its own line.


---

## 2. Repository Structure

Important ones:
- `_/data/federated_learning/<name>/` contains a versioned `Trace` layout created by the sparse runner for deterministic protocol replay.

Rest:
- `FL_code/` contains the core of the code, with FL logic, and also the main body of protocols, codec, quantizer and wyner-ziv logic and training.
- `Experiments/` A directory with other folders where tests, checks and small scale runs are experimented with.

- `Thesis_doc/` contains the thesis LaTeX source. ignore this
- `Paper_doc/` contains paper LaTeX source and related figures. ignore this

---

## 3. Checks when writing
1. Does this need to exist?   → no: skip it (YAGNI)
2. Already in this codebase?  → reuse it, don't rewrite
3. Stdlib does it?            → use it
4. Native platform feature?   → use it
5. Installed dependency?      → use it
6. Fit in one line?           → one line
7. Only then: the minimum that works

## Long experiments

- Run multi-hour jobs through the detached experiment supervisor, never an interactive tool session. It records health at 1, 5, and 30 minutes, then every 90 minutes; rely on those checks instead of babysitting with repeated sleeps. Bound parallel cases by measured host RAM so the job survives chat/session loss.
- Keep its live plotter enabled: it refreshes changed results in disposable processes, survives plot-code edits without replacing the last good plots, and exits with the experiment.

### Protocol runs and result reporting

- Use a fresh named plan/output; launch with `.venv/bin/python -m Experiments.Protocol_test.detached_replay launch <plan.json>`. Preserve the trace device for matched comparisons; stop via the specific `systemctl --user stop <unit>`, never `pkill`. See `Experiments/Protocol_test/README.md` for details.
- Default comparison: all matched clients over **one trained + nine frozen rounds**, per start and pooled. Average cumulative rates and raw WMAPE before `20*log10(mean_wmape/100)` and `rd_metrics.rd_curve_gap`; never average individual gap percentages. Label partial windows and trained/frozen diagnostic splits explicitly.
- Report a **full square text table of equal-distortion rate changes (%)**, row versus column, negative favoring the row; recompute reverse entries. Full-SI rANS is primary; native transport is separate. Show with/without symmetric whole-window client exclusions, their reasons and retained coverage.
- Alongside it, show stage RD tables in bits/float: **marginal, joint/pre-finetuning prior, temporal/full-SI fine-tuned priors, native/temporal/full-SI rANS, and WMAPE dB (raw %)**. Scalar totals appear only at the final stage; missing measurements are `—`, not inferred. Frozen prior rates evaluate fixed trained weights on refreshed target/SI, not another fit; state any actual refitting and flag dead/blown stages or symbol errors.

---

## 4. Code Review Goals

- Logic and behavior are important review concerns and must be checked carefully.
- Optimize for code that human researchers can quickly read, understand, and modify without reconstructing hidden intent.
- Keep a clear structure with proper separation of concerns and tasks. A file, class, or function's place and boundaries should make its responsibility apparent.
- Prefer a direct data flow and descriptive names over extra branches, hidden state changes, duplicate logic, or speculative flexibility.
- Add functions and abstractions when they clarify a distinct task, protect a meaningful boundary, or remove real duplication; avoid both monolithic code and needless one-use indirection.
- Keep research assumptions and important state transitions explicit in the closest relevant code, using concise comments or docstrings where the structure alone cannot explain them.
- Recheck criticism against the surrounding code and tests before reporting it, separating demonstrated problems from optional style preferences.
