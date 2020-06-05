#!/bin/bash

#OMP_NUM_THREADS=12 python3 -u stage1_active_pref_learning.py LR 0 duc04_lr_10inter "[random,unc]" . 2 DUC2004 10
OMP_NUM_THREADS=12 python3 -u stage1_active_pref_learning.py GPPLHH 0 duc04_gpplhh_10inter_random "[random]" . 2 DUC2004 10
#python3 -u stage1_active_pref_learning.py GPPLHH 0 duc04_gpplhh_10inter "[random,pair_unc,eig,tig,imp]" . 1 DUC2004 10
OMP_NUM_THREADS=12 python3 -u stage1_active_pref_learning.py GPPLH 0 duc04_gpplh_10inter "[random,pair_unc]" . 2 DUC2004 10
OMP_NUM_THREADS=12 python3 -u stage1_active_pref_learning.py GPPL 0 duc04_gppl_10inter "[random,pair_unc]" . 2 DUC2004 10



