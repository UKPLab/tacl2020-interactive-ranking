#!/bin/bash

OMP_NUM_THREADS=12 python3 -u stage1_active_pref_learning.py GPPLHH 0 duc02_gpplhh_10inter "[random,pair_unc,eig,tig,imp]" . 2 DUC2002 10
OMP_NUM_THREADS=12 python3 -u stage1_active_pref_learning.py GPPLH 0 duc02_gpplh_10inter "[random,pair_unc]" . 2 DUC2002 10
OMP_NUM_THREADS=12 python3 -u stage1_active_pref_learning.py GPPL 0 duc02_gppl_10inter "[random,pair_unc]" . 2 DUC2002 10
OMP_NUM_THREADS=12 python3 -u stage1_active_pref_learning.py LR 0 duc02_lr_10inter "[random,unc]" . 2 DUC2002 10

