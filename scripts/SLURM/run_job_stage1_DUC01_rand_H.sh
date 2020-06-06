#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J s101Hrand
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL
#SBATCH -e /work/scratch/es14puve/s101Hrand.err.%j
#SBATCH -o /work/scratch/es14puve/s101Hrand.out.%j
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=16384
#SBATCH --exclusive
#SBATCH -C avx

# ----------------------------------

module load intel python/3.6.8

python3 -u stage1_active_pref_learning.py GPPLH 0 duc01_rand_H "[random]" /work/scratch/es14puve 24

# to run the profiler. Note that this will mean we get no .out logs because output is diverted into stage1.profile.
#python3 -m cProfile -o stage1.profile stage1_active_pref_learning.py GPPL 0 /work/scratch/es14puve
