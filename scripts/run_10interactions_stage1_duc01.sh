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

module load intel python/3.6.8

export OMP_NUM_THREADS=12

python3 -u stage1_active_pref_learning.py GPPLHH 0 duc01_gpplhh_10inter2 "[random,pair_unc,eig,tig,imp]" . 2 DUC2001 10
python3 -u stage1_active_pref_learning.py GPPLH 0 duc01_gpplh_10inter2 "[random,pair_unc]" . 2 DUC2001 10
python3 -u stage1_active_pref_learning.py GPPL 0 duc01_gppl_10inter2 "[random,pair_unc]" . 2 DUC2001 10
python3 -u stage1_active_pref_learning.py LR 0 duc01_lr_10inter2 "[random,unc]" . 2 DUC2001 10



