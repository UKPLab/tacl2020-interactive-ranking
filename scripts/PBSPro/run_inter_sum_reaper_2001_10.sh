#!/bin/sh

# Run the interactive summarisation simulations with SUPERT

# Job name
#PBS -N intsum_rea_01

# Output file
#PBS -o pbs_intersumrea2001_10_output.log

# Error file
#PBS -e pbs_intersumrea2001_10_err.log

# request resources and set limits
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=24:mem=128GB:ompthreads=24
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/pytorch

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /work/es1595/text_ranking_bayesian_optimisation

# Run the script using heuristics only and no interactions.
python -u stage1_active_pref_learning.py H 0 duc01_reaper_H "[random]" . 4 DUC2001 0 april

#  run the script for each DUC dataset with GPPL-IMP, GPPL-UNPA, GPPL-EIG, GPPL-Random, BT-Random.
#python -u stage1_active_pref_learning.py GPPLHH 0 duc01_reaper_imp_gpplhh_10 "[imp]" . 4 DUC2001 10 april
#python -u stage1_active_pref_learning.py LR     0 duc01_reaper_ran_lr_10     "[random]" . 4 DUC2001 10 april
#python -u stage1_active_pref_learning.py GPPLHH 0 duc01_reaper_ran_gpplhh_10 "[random]" . 4 DUC2001 10 april
#python -u stage1_active_pref_learning.py GPPLHH 0 duc01_reaper_unp_gpplhh_10 "[pair_unc]" . 4 DUC2001 10 april
#python -u stage1_active_pref_learning.py GPPLHH 0 duc01_reaper_eig_gpplhh_10 "[eig]" . 4 DUC2001 10 april

# To submit: qsub run_bert_cqa.sh
# To display the queue: qstat -Q gpu (this is usually where the GPU job ends up)
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>

# To monitor job progress:
# qstat -f | grep exec_host
# Find the node where this job is running.
# ssh to the node.
# tail /var/spool/pbs/spool/<job ID>.bp1.OU
