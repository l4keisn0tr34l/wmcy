# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## GPU facts

Verified local hardware/runtime:

```text
GPU: NVIDIA GeForce RTX 3050 Mobile, 4 GiB
Driver: 595.84
Driver-reported CUDA compatibility: 13.2
Current torch: 2.9.1+cpu (CUDA unavailable only because CPU wheel installed)
CPU: 16 logical cores; RAM 15 GiB
Disk available: ~21 GiB after CUDA metadata probe
```

PyTorch index confirms `torch==2.9.1+cu128` exists and is compatible with the current driver. No system CUDA toolkit or sudo should be needed because the wheel supplies runtime libraries.

## Current batch: device-portable RSSM

1. Add shared `auto|cpu|cuda` device resolution with clear CUDA errors and deterministic seed handling.
2. Add `--device auto` to active flattened/graph training and V3/public training paths.
3. Move models, batches, permutation indices, positive weights, smoke tests, validation, Monte Carlo forecasting, and equivariance audits consistently to the chosen device.
4. Keep checkpoints device-portable via CPU state dictionaries/map-location.
5. Preserve CPU compatibility and old CLI defaults through `auto` fallback.
6. Add separate CPU/CUDA requirement files; install `torch==2.9.1+cu128` into `.venv` without sudo.
7. Run CUDA availability, forward/backward, exact graph equivariance, CPU-vs-GPU deterministic-output tolerance, and VRAM tests.
8. Run a short selection-only training benchmark without reopening V3 test or changing published models.

## Expected limitation

The current 358k-parameter graph model and batches of 32 are small. GPU may provide modest rather than dramatic speedup because Python recurrent loops, six seeds/regimes, early stopping, data loading, and metric code dominate. Branching models/public batches should benefit more. PCAP parsing remains CPU/disk-bound.
