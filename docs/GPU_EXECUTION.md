# GPU Execution

## Verified environment

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
VRAM: 4 GiB (3.94 GB reported to PyTorch)
Driver: 595.84
PyTorch: 2.9.1+cu128
Compute capability: 8.6
CUDA available: yes
```

The CUDA wheel bundles user-space runtime libraries; no system CUDA toolkit or sudo installation was required.

Install/reproduce:

```bash
.venv/bin/python -m pip install -r requirements-rssm-cu128.txt
```

The original `requirements-rssm.txt` remains the explicit CPU installation for machines without NVIDIA hardware.

## Device selection

Primary RSSM scripts accept:

```text
--device auto   use CUDA when available, otherwise CPU (default)
--device cuda   require CUDA and fail clearly if unavailable
--device cpu    force CPU
```

Examples:

```bash
.venv/bin/python scripts/30_train_graph_rssm_v3.py \
  --device cuda --batch-size 128 --selection-only

.venv/bin/python scripts/28_pretrain_graph_rssm_unsw.py \
  --device cuda --public-batch-size 128 --lab-batch-size 64
```

Do not rerun completed V3 test evaluation merely to benchmark hardware. `--selection-only` exits before loading test.

## Verified numerical and memory gates

`scripts/32_test_torch_devices.py` loads V3 train only and checks deterministic parity, backward gradients, graph equivariance, speed, and peak memory.

At batch 128:

```text
CPU/CUDA deterministic maximum delta: 7.45e-08
CUDA graph-equivariance maximum delta: 2.24e-08
finite backward gradients:             pass
CUDA peak allocated memory:            202 MB
CUDA speedup over CPU per step:         1.78×
```

The full CUDA validation-only V3 training path also passed without loading test.

## Performance qualification

The current model is only about 358k parameters and uses short Python-level recurrent loops. GPU launch/transfer overhead dominates small batches:

```text
batch 32 speedup:  0.86× (GPU slightly slower)
batch 128 speedup: 1.78× per optimization step
```

Batch 128 also reduces the number of steps per epoch, so it should materially shorten future larger experiments. Changing batch size changes optimization behavior and must be declared before validation—not used to recreate already reported models.

GPU acceleration applies to model training, Monte Carlo rollout, and equivariance evaluation. PCAP decoding, CSV processing, pandas aggregation, and graph-state construction remain CPU/disk-bound.

## Checkpoints

New training paths move checkpoint state dictionaries to CPU before saving. Consequently, checkpoints trained on CUDA remain loadable on CPU-only machines.
