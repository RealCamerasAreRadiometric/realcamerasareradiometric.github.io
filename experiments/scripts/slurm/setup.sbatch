#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root (`sbatch experiments/scripts/slurm/setup.sbatch`, or `bash experiments/scripts/reproduce/reproduce_results_vcai.sh`); logs go to ./logs/,
# which must exist before submission (Slurm opens the log before the script runs; the runner creates it).
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --job-name=rc-setup
#SBATCH --output=logs/rc_setup_%j.log
#SBATCH --error=logs/rc_setup_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# setup.sbatch — the experiments environment, once, before every other job of a launch (they depend on it: --dependency=afterok).
#
# Two environments, each with its own untracked Manifest: experiments (the drivers) and the package itself (bin/refine.jl runs in
# --project=RadiometricCamera). Instantiate resolves each from the General registry, then both are precompiled
# with CUDA and the GPU extensions loaded, at the jobs' own flags. CPU only (dgx01's
# rule, 2026-11-13: GPU tasks never compile): loading CUDA and its extensions needs no GPU, so every GPU task that follows only
# loads and runs. CUDA's runtime artifacts are fetched by the first GPU task that initialises the device.
#
# RESOURCES: the CPUs, the memory per CPU (8480M, the node ratio) and the GPU are the runner's — configs/reproduce.toml [[stage]]
# cpus_per_task, mem_per_cpu and gpu, the one source; a launcher submitted by hand states them on its sbatch line.
# dependency chain (a 32-CPU precompile ran at ~5 cores, dgx01 2026-11-13).
# =============================================================================

echo "[$(date +%T)] julia=$("$JULIA" --version)  data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  (CPU only)"
"$JULIA" --startup-file=no --project=experiments -e 'using Pkg; Pkg.Registry.update(); Pkg.instantiate(); Pkg.precompile()'
"$JULIA" --startup-file=no --project=RadiometricCamera -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
"$JULIA" --startup-file=no --project=experiments -e 'import CUDA, RadiometricCamera, PanoramaStitching'   # the GPU extensions compiled, no GPU
"$JULIA" --startup-file=no --project=RadiometricCamera -e 'import CUDA, RadiometricCamera'
echo "[$(date +%T)] setup done"
