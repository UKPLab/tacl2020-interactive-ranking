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

python3 -u stage1_coala.py GPPLHH 0 lno03_gpplhh_1inter "[random,pair_unc,eig,imp,tp]" . 2 1
python3 -u stage1_coala.py LR 0 lno03_lr_1inter "[random,unc]" . 2 1

python3 -u stage1_coala.py GPPLHH 0 lno03_gpplhh_3inter "[random,pair_unc,eig,imp,tp]" . 2 3
python3 -u stage1_coala.py LR 0 lno03_lr_3inter "[random,unc]" . 2 3

python3 -u stage1_coala.py GPPLHH 0 lno03_gpplhh_5inter "[random,pair_unc,eig,imp,tp]" . 2 5
python3 -u stage1_coala.py LR 0 lno03_lr_5inter "[random,unc]" . 2 5

python3 -u stage1_coala.py GPPLHH 0 lno03_gpplhh_7inter "[random,pair_unc,eig,imp,tp]" . 2 7
python3 -u stage1_coala.py LR 0 lno03_lr_7inter "[random,unc]" . 2 7

python3 -u stage1_coala.py GPPLHH 0 lno03_gpplhh_10inter "[random,pair_unc,eig,imp,tp]" . 2 10
python3 -u stage1_coala.py LR 0 lno03_lr_10inter "[random,unc]" . 2 10

python3 -u stage1_coala.py GPPLHH 0 lno03_gpplhh_15inter "[random,pair_unc,eig,imp,tp]" . 2 15
python3 -u stage1_coala.py LR 0 lno03_lr_15inter "[random,unc]" . 2 15

python3 -u stage1_coala.py GPPLHH 0 lno03_gpplhh_20inter "[random,pair_unc,eig,imp,tp]" . 2 20
python3 -u stage1_coala.py LR 0 lno03_lr_20inter "[random,unc]" . 2 20

python3 -u stage1_coala.py GPPLHH 0 lno03_gpplhh_25inter "[random,pair_unc,eig,imp,tp]" . 2 25
python3 -u stage1_coala.py LR 0 lno03_lr_25inter "[random,unc]" . 2 25

#python3 -u stage1_coala.py GPPLHH 0 lno03_gpplhh_50inter "[random,pair_unc,eig,imp]" . 2 50
#python3 -u stage1_coala.py LR 0 lno03_lr_50inter "[random,unc]" . 2 50
#
#python3 -u stage1_coala.py GPPLHH 0 lno03_gpplhh_100inter "[pair_unc,eig,imp,random]" . 2 100
#python3 -u stage1_coala.py LR 0 lno03_lr_100inter "[random,unc]" . 2 100


