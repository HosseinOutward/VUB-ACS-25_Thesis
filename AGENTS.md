# Project Context

This repository supports PhD-level research into gradient compression for federated learning using distributed source coding, especially learned Wyner-Ziv methods.

The core research direction is a two-step compression pipeline:

1. A distributed source coding quantizer, preferably a learned Wyner-Ziv quantizer.
2. A conditional entropy coder, ideally based on Slepian-Wolf coding.

The central design question is always: what side information is available at the decoder, and can it legitimately be used as a prior?

---

## 0. Research Principles

- **Two-step compression.** Keep quantization and entropy coding conceptually separate. The quantizer produces symbols; the conditional entropy coder compresses those symbols using decoder-available side information.
- **Decoder-side priors matter.** Both the DSC quantizer and the conditional entropy coder need priors. Before using a prior, identify where decoding happens and which data is actually available there. Do not assume worker-only or server-only information is available everywhere.
- **Multi-stage learned quantization per float.** A learned quantizer may emit several vectors of indices per float. Each stage can use earlier stage outputs as side information for later stages, producing multiple low-bit symbols from one floating-point value.
- **Staging may extend beyond single floats.** The same staged logic may later flow across broader quantization and coding stages. The exact method is still a research question, so keep implementations explicit and easy to inspect.
- **Secrets never touch saved artifacts.** Configuration files store the name of an environment variable, never the secret value itself.
- **Structural validation only.** Pydantic validates types, formats, enums, and required fields. Do not add fabrication checks, provenance checks, length checks, or page-count checks.

---

## 1. Coding Standards

These standards apply to every module unless a more specific local instruction says otherwise.

- **Use full type hints** on every function signature and class attribute.
- **Use advanced python syntax**, Its encouraged to use python best practices and advanced syntax to improve the code logic and readability. avoid over engineering for the sake of syntax though. for example, `pathlib.Path` for filesystem paths over raw `str`, and many other uses of tools given by python and common libraries.
- **Keep code to a minimum** Common thread among below standards too, is to make use of dynamic approaches, judgment of if an extra code had any real impact or use, and if an extra function and class needed to be made. Less is always better mostly since it helps with readibilty and debugging. all else is secondary.
- **Write concise docstrings.** Public functions need a one-paragraph docstring. Each shared data contract needs a class-level docstring explaining what it represents and who produces and consumes it.
- **Comments explain why, not what.** Comment only when intent is non-obvious, such as a conditional field requirement or cache ordering. Do not narrate self-evident code.
- **Fail loudly and clearly.** Hidden behavior is worse than a visible failure. On error, raise with a message that names the offending file, field, or stage. Avoid silent fallbacks, (backward) compatibility logic, and unnecessary recovery logic.
- **No bare `except`.** Catch specific exceptions only when handling them is useful. Use assertions for internal assumptions.
- **rasising errors** Instead of if-raises, use asserts where its a valid replacement. also only use the asserts where either an assumption has to be checked, or some future error will happen which might not be clear. for the latter, this means avoid asserts just to make a custom text for fail, when its already obvious. instead use it where downstream fails are not clear or worse, not even caught.
- **No global mutable state.** Pass `Config` and other dependencies explicitly. Do not introduce module-level mutable singletons.
- **Avoid needless abstraction.** Do not add tiny one-use functions or renaming-only helpers. Add abstractions only when they remove real duplication or make a complex concept easier to reason about.
- **Keep control flow shallow.** Avoid unnecessary nested loops and conditionals. Prefer direct, readable logic.
- **Avoid meaningless intermediate variables.** Create new names only when they clarify a non-trivial transformation or prevent repeated complex expressions.

---

## 2. Repository Structure

- `FL_code/` contains the federated learning and compression implementation.
- `FL_code/other_protocols/` contains alternative or baseline protocol implementations.
- `FL_code/experiments/` contains notebooks and scripts for analysis, diagnostics, and experiment setup.
- `FL_code/records/` contains experiment outputs and run configurations.
- `FL_code/docs/` contains implementation notes and protocol documentation.
- `FL_code/data/` contains local data assets and pretrained model files used by experiments.
- `Thesis_doc/` contains the thesis LaTeX source.
- `Paper_doc/` contains paper LaTeX source and related figures.

---

## 3. Checks when writing
1. Does this need to exist?   → no: skip it (YAGNI)
2. Already in this codebase?  → reuse it, don't rewrite
3. Stdlib does it?            → use it
4. Native platform feature?   → use it
5. Installed dependency?      → use it
6. Fit in one line?           → one line
7. Only then: the minimum that works