# Sparse initial-side-information study

This study compares temporal (`T`) and oracle/server (`O`) learned Wyner–Ziv quantization on one
shared 30-client ResNet-18/SVHN raw-FL trace. Independent windows start at rounds 5, 10, and 20.
Each window runs one trained round followed by six frozen rounds. Every O/T and start-round
placement is measured twice: once from raw initial SI and once from classical two-split SI.

The five rounds before each start seed decoder history with raw client updates or with their
equal-frequency two-split reconstructions. From the start onward, every selected client keeps the
reconstruction produced by decoding, so frozen rounds consume the protocol's evolving decoded
history. Compression never affects the server model used to generate the raw trace.

- `experiment/`: the reviewed plan and FL configuration.
- `runs/resnet18_svhn_30c/`: generated records and analysis (ignored by git).
- Bulk trace and compressed initial-history caches: `/mnt/nvme1/FL_WZ_Quant/sparse_initial_si/`.

Each start and SI mode is cached once under `protocol_cache/initial_history/` as actual
`CompressedReconstructionHistory` dictionary/code chunks. Cases load an independent copy; they do
not rebuild SI or convert the cache to dense `HistoryVector` objects.
Quantizer and prior fitting are never persistently cached. The O/T round fits both the quantizer and
the separate retrained conditional prior once. `F` reuses both frozen models while refreshing their
SI input from evolving decoded history; it does not refit either model.

The quantizer settings use three binary planes,
`hidden_dim=256`, 20 epochs, 200,000 training samples, three quantizer candidates, and one prior
repeat. Seed 43 is explicit on both training rounds.

N=2 comes from the uncached three-plane probe in `experiment/quantizer_bin_occupancy.json`: on
T/start5/client0, joint codewords 6 and 7 contained 1,866,072 and 9,315,570 coordinates; all other
joint codewords contained zero. The 32-sample minimum therefore retains two real bins.

Run status: not started; the earlier incomplete outputs were removed after the replay implementation
changed. Cases are queued O then T for raw SI, followed by O then T for two-split SI, at each start.
Incomplete previews use `_INCOMPLETE` in every image filename and title. Final plotting should run
after a successful replay exit.
