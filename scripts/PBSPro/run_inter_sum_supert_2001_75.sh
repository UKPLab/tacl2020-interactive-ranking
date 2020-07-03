#!/bin/sh

# Run the interactive summarisation simulations with SUPERT

# Job name
#PBS -N intsum_s75_01

# Output file
#PBS -o pbs_intersumsup2001_75_output.log

# Error file
#PBS -e pbs_intersumsup2001_75_err.log

# request resources and set limits
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=24:mem=128GB:ompthreads=24
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/pytorch

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /work/es1595/text_ranking_bayesian_optimisation

# Run the script using heuristics only and no interactions.

#  run the script for each DUC dataset with GPPL-IMP, GPPL-UNPA, GPPL-EIG, GPPL-Random, BT-Random.
python -u stage1_active_pref_learning.py GPPLHH 0 duc01_supert_bi_eig_gpplhh_75 "[eig]" . 4 DUC2001 75 supert 200 0.5 results_ls05 1
python -u stage1_active_pref_learning.py GPPLHH 0 duc01_supert_bi_imp_gpplhh_75 "[imp]" . 4 DUC2001 75 supert 200 0.5 results_ls05 1
python -u stage1_active_pref_learning.py LR     0 duc01_supert_bi_ran_lr_75     "[random,unc]" . 4 DUC2001 75 supert 200 0.5 results_ls05 1
python -u stage1_active_pref_learning.py GPPLHH 0 duc01_supert_bi_ran_gpplhh_75 "[random]" . 4 DUC2001 75 supert 200 0.5 results_ls05 1

# To submit: qsub run_bert_cqa.sh
# To display the queue: qstat -Q gpu (this is usually where the GPU job ends up)
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>

# To monitor job progress:
# qstat -f | grep exec_host
# Find the node where this job is running.
# ssh to the node.
# tail /var/spool/pbs/spool/<job ID>.bp1.OU
