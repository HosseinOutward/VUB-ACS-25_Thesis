# TR prior provenance check

`check.py` runs the real `wz_cancer` protocol with a TR round on synthetic 200k-element deltas for
three clients (rounds 0-4 as history, TR at round 5, F at round 6) and logs what each quantizer and
prior fit receives: client 0 is AR(1) on its own past, client 1 fresh noise, client 2 a copy of
client 0's same-round delta.

Result (2026-09-29, former 6-layer defaults with the token's small-epoch overrides): TR's SI is
exactly the client's own five previous reconstructions (the F round slides it to rounds 1-5); the
native-prior fine-tune fits this round's symbols with those same five SI columns; with useless SI
(client 1) the fine-tuned prior lands on the SI-free floor, the empirical 3-stage symbol entropy
(1.625 vs 1.626 bit), so nothing leaks; and only the server-SI comparison prior can exploit client
0's same-round update (client 2: 0.764 temporal vs 0.481 server). The low fine-tuned TR rate on real
data is genuine temporal correlation, not leakage.

```bash
PYTHONPATH=. .venv/bin/python Experiments/tr_prior_provenance_check/check.py
```
