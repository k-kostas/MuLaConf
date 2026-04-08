"""
Memory Management Constants for MuLaConf.

This file contains the core memory and batching thresholds used by the
InductiveConformalPredictor. Because the Powerset Scoring approach scales
exponentially O(2^C), where C is the number of labels, these constants protect the system from Out-Of-Memory (OOM)
crashes by throttling the batching engine.

Users can modify these values to optimize performance based on their specific
CPU/GPU hardware limits or the controlled enabling or disabling of safety mechanisms
for benchmarking.


=====================
APPROXIMATION PROCESS
=====================
1. Heavy Matrix Math (Scoring & Penalties - RAM & VRAM)
   During `all_combinations_scoring`, PyTorch creates massive 3D float32 tensors.
   A single `float32` number consumes 4 bytes. However, PyTorch must hold multiple
   intermediate tensors simultaneously (combinations, probabilities, subtraction
   results, and absolute errors) to execute the operation. We apply a 4x multiplier
   (16 bytes total) to account for this overhead.

   Formula: Memory ≈ max_combinations * n_classes * 16 bytes

2. Prediction Extraction (Region Building)
   During the final extraction phase, no heavy matrix math occurs. PyTorch only
   loads a flat array of pre-calculated p-values (`float32` = 4 bytes) and creates
   a boolean mask (`True/False` = 1 byte) to filter them.

   Formula: Memory ≈ batch_size * 5 bytes


=========================================
GPU VRAM CHEAT SHEET (For heavy scoring)
=========================================
Adjust `_GPU_MAX_COMBINATIONS` based on your total VRAM.

* 4GB - 6GB VRAM  (Older laptops, GTX 1650)   : 3_000_000
* 8GB VRAM        (Standard GPUs, RTX 3060)   : 5_000_000  (Default)
* 12GB - 16GB VRAM(High-end, RTX 4080)        : 10_000_000
* 24GB+ VRAM      (Enthusiast, RTX 4090/A100) : 15_000_000


==================================================
SYSTEM RAM CHEAT SHEET (For CPU Math & Extraction)
==================================================
Adjust `_CPU_MAX_COMBINATIONS` (heavy math) and `_REGION_BATCH_SIZE` (lightweight filtering)
based on your total System RAM. The CPU math default is strictly lower than the GPU
default to leave memory allocation for the Operating System and the original dataset.

* 8GB RAM:
    _CPU_MAX_COMBINATIONS = 2_000_000  (Default)
    _REGION_BATCH_SIZE    = 50_000_000 (Default)

* 16GB RAM:
    _CPU_MAX_COMBINATIONS = 5_000_000
    _REGION_BATCH_SIZE    = 100_000_000

* 32GB RAM:
    _CPU_MAX_COMBINATIONS = 12_000_000
    _REGION_BATCH_SIZE    = 250_000_000

* 64GB+ RAM:
    _CPU_MAX_COMBINATIONS = 30_000_000
    _REGION_BATCH_SIZE    = 500_000_000


======================
BENCHMARKING CONTROLS
======================
Prevents VRAM fragmentation and Out-Of-Memory (OOM) crashes on standard hardware.
Default: True (Safety First)


When to set to False (Raw Performance Benchmarking):
- `torch.cuda.empty_cache()` forces the GPU to halt, synchronize, and communicate
  with the OS to release memory blocks. This OS-level communication adds a delay to operations,
  artificially inflating timing benchmarks.
- If you are running rigorous execution-time experiments (and have sufficient VRAM),
  set this to False to expose the true, uninterrupted speed of the GPU math.
"""

# The maximum number of label combinations the CPU can process in a single batch
# during heavy 3D tensor matrix multiplications (Scoring & Penalties).
# Safe default for minimum-spec laptops (8GB+ RAM): 2 million combinations.
_CPU_MAX_COMBINATIONS = 2_000_000

# The maximum number of label combinations the GPU can process in a single batch
# during heavy 3D tensor matrix multiplications (Scoring & Penalties).
# Safe default for modern GPUs (8GB+ VRAM): 5 million combinations.
_GPU_MAX_COMBINATIONS = 5_000_000

# Caps the memory usage when extracting the final prediction sets (System RAM).
# Because this relies on simple 1D boolean array filtering rather than heavy
# matrix math, this limit can be significantly higher than the calculation limits.
# Safe default: 50 million combinations.
_REGION_BATCH_SIZE = 50_000_000

# Controls whether the engine aggressively clears the CUDA memory cache
# after executing heavy 3D tensor math (Powerset Scoring and Penalties).
_EMPTY_CUDA_CACHE = True