#!/bin/sh

# Run the interactive summarisation simulations with SUPERT

# Job name
#PBS -N intsum_sup_01

# Output file
#PBS -o pbs_intersumsup2001_output.log

# Error file
#PBS -e pbs_intersumsup2001_err.log

# request resources and set limits
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=24:mem=128GB:ompthreads=12
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/pytorch

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /work/es1595/text_ranking_bayesian_optimisation

# Run the script using heuristics only and no interactions.
#python -u stage1_active_pref_learning.py H 0 duc01_supert_H "[random]" . 2 DUC2001 0 supert

#  run the script for each DUC dataset with GPPL-IMP, GPPL-UNPA, GPPL-EIG, GPPL-Random, BT-Random.
python -u stage1_active_pref_learning.py GPPLHH 0 duc01_supert_imp_gpplhh "[imp]" . 2 DUC2001 100 supert
python -u stage1_active_pref_learning.py LR     0 duc01_supert_ran_lr     "[random]" . 2 DUC2001 100 supert
python -u stage1_active_pref_learning.py GPPLHH 0 duc01_supert_ran_gpplhh "[random]" . 2 DUC2001 100 supert
python -u stage1_active_pref_learning.py GPPLHH 0 duc01_supert_unp_gpplhh "[pair_unc]" . 2 DUC2001 100 supert
python -u stage1_active_pref_learning.py GPPLHH 0 duc01_supert_eig_gpplhh "[eig]" . 2 DUC2001 100 supert

# To submit: qsub run_bert_cqa.sh
# To display the queue: qstat -Q gpu (this is usually where the GPU job ends up)
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>

# To monitor job progress:
# qstat -f | grep exec_host
# Find the node where this job is running.
# ssh to the node.
# tail /var/spool/pbs/spool/<job ID>.bp1.OU
