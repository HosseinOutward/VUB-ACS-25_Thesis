# Refactor TODO

### 1. Make side-information ownership explicit

- **Scope:** `FL_code/cancer_protocol.py`, `FL_code/cancer_quantizer.py`, `FL_code/prior_calculator.py`
- **Why:** Protocol state currently mixes client-side histories, server-side histories, prior-training side information, and quantizer side information in mutable lists. 
- **Risk:** The issue with this is possible room to mix them up and cause bugs. the benfit is that the data is accessible for steps which are not being experimented on but need more data like the SW step.
- **Acceptance:** Each side-information tensor has an explicit owner, producer, consumer, 

### 3. Audit prior-model training sampling

- **Scope:** `FL_code/prior_calculator.py`
- **Why:** `_train_prior_model` samples random indices for `bins_subset` and `si_subset`, but the one-hot code construction should be reviewed carefully against the sampled batch.
- **Risk:** A subtle index mismatch could train priors on bins that do not correspond to the side information batch.
- **Acceptance:** Add a small deterministic test that proves sampled bins and sampled side information remain aligned. a simple assert should do it.

### 4. Replace ad hoc configs with Pydantic v2 contracts

- **Scope:** `FL_code/run_fl.py`, `FL_code/cancer_protocol.py`, experiment entrypoints
- **Why:** Dataclasses are convenient, but the project guidelines call for Pydantic v2 when structured configuration contracts are needed.
- **Risk:** Config drift can silently change experimental meaning across scripts.
- **Acceptance:** Config models validate structure only, use `Path` for paths, and store environment variable names rather than secret values.

### 6. Normalize import/package layout

- **Scope:** `FL_code/`, `FL_code/other_protocols/`
- **Why:** Some modules use direct imports while alternate protocols patch `sys.path` for fallback imports.
- **Risk:** Running from the repository root, from `FL_code/`, or via `python -m` can produce different import behavior.
- **Acceptance:** One supported invocation style is documented and imports work without runtime `sys.path` mutation.

### 7. Strengthen metric edge cases

- **Scope:** `FL_code/codec.py`, `FL_code/utils.py`
- **Why:** Compression metrics assume `model_size` is present, weighted pesrcentage denominators are non-zero, and AUC is defined for the current labels.
- **Risk:** Short debug runs or unusual client partitions can fail late with unclear errors.
- **Acceptance:** Errors name the missing metric input or undefined evaluation condition; no silent fallback metrics are introduced. but nothing more, and through asserts

### 8. Add focused verification scripts

- **Scope:** new tests or small scripts near `FL_code/experiments/`
- **Why:** Current behavior is research-heavy and GPU-heavy, so full runs are expensive.
- **Risk:** Refactors can break serialization, flatten/unflatten, payload shape, or side-information alignment without immediate signal.
- **Acceptance:** Add cheap checks for serialization round-trip, `StateDictManager` flatten/unflatten, codec factory errors, and prior shape/rate calculations.
