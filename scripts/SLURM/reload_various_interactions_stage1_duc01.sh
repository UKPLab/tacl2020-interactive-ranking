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
##SBATCH -C avx

# ----------------------------------

module load intel python/3.6.8

export OMP_NUM_THREADS=24

python3 -u reload_stage1_results.py GPPLHH 0 duc01_gpplhh_10inter2 "[random,pair_unc,eig,imp]" . 2 DUC2001 10
python3 -u reload_stage1_results.py LR 0 duc01_lr_10inter2 "[random,unc]" . 2 DUC2001 10

python3 -u reload_stage1_results.py GPPLHH 0 duc01_gpplhh_20inter2 "[random,pair_unc,eig,imp]" . 2 DUC2001 20
python3 -u reload_stage1_results.py LR 0 duc01_lr_20inter2 "[random,unc]" . 2 DUC2001 20

python3 -u reload_stage1_results.py GPPLHH 0 duc01_gpplhh_50inter2 "[random,pair_unc,eig,imp]" . 2 DUC2001 50
python3 -u reload_stage1_results.py LR 0 duc01_lr_50inter2 "[random,unc]" . 2 DUC2001 50

python3 -u reload_stage1_results.py GPPLHH 0 duc01_gpplhh_75inter2 "[random,pair_unc,eig,imp]" . 2 DUC2001 75
python3 -u reload_stage1_results.py LR 0 duc01_lr_75inter2 "[random,unc]" . 2 DUC2001 75

python3 -u reload_stage1_results.py GPPLHH 0 duc01_gpplhh_100inter2 "[random,pair_unc,eig,imp]" . 2 DUC2001 100
python3 -u reload_stage1_results.py LR 0 duc01_lr_100inter2 "[random,unc]" . 2 DUC2001 100


