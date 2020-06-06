#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J s101eig
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL
#SBATCH -e /work/scratch/es14puve/s101eig.err.%j
#SBATCH -o /work/scratch/es14puve/s101eig.out.%j
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=16384
#SBATCH --exclusive
#SBATCH -C avx

# ----------------------------------

#module load intel python/3.6.8

export OMP_NUM_THREADS=24

# What do these options mean?
# Arg1: interactie method. H=heuristic only?
# Arg2: no. debug samples. 0=don't debug, use all data.
# Arg3: output folder name
# Arg4: querier types
# Arg 5: data root directory
# Arg 6: no. threads
# Arg 7: no. interaction rounds
# Aim: run the pure COALA setup with no interactions.
python3 -u stage1_coala.py H 0 lno03_coala_nointer "[random]" . 2 0



