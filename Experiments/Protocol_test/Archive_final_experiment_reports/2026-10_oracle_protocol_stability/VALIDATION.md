# Closure checks — 2026-10-04

The retained configuration and cleanup were checked without launching another long experiment.

- `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q`: **23 passed**.
  Nine acceptance/identity tests plus fourteen current plan/record/default contract checks.
- All seven retained plan files load; all their cases construct without archived registrations.
- `wz_cancer_multi`, `wz_cancer_multi_warm`, and `wz_cancer_confirmation` fail as unregistered
  in a fresh process importing only the active runner.
- Stable layers/lr/epochs and derived warm35/lr×0.3/tau0.5 are unchanged. Ordinary quantizer
  rejection allowance remains two slots; the experimental six-slot allowance is archived.
- Each of the six original OR/TR case CSVs still validates against ITS ORIGINAL stored identity:
  300 rows each, 1800 total. No raw output or checkpoint was deleted or rewritten.
- `git diff --check` passed. Historical source/evidence snapshots are retained under this archive.
- No quantizer service remained running. The idle global plotter was stopped with its exact
  user systemd unit as part of closing the investigation; future launch instructions restart it.

An initial broad pytest discovery also picked up the pre-existing, gitignored `archived_tests/`
directory: 54 passed and12 failed. Eleven failures concern removed legacy APIs/config fields
or old warm-record suffixes (`conditional_state_dict`, `make_trained_prior_model`, protocol-level
`warm_retrain_repeats`, and old provenance fields). These expectations already disagree with
the baseline source. One mocked cold-prior test performs no training on constant symbols:
its untrained random prior correctly fails the new marginal gate. Those local historical tests
were not modified or restored as supported tests. Routine discovery now excludes archive folders,
including `archived_tests/`; current suffix and marginal-gate behavior have dedicated live tests.

Archived experiment tests and modules preserve their original pre-cleanup implementation.
They are not expected to run against cleaned production support: reproduce in an isolated
historical-layout worktree using the source snapshot described in README.md.

The branch is merged into local master and deleted only after successful fast-forward merge.
No push, remote reset/reconciliation, raw-result commit, or new experiment launch is performed.
