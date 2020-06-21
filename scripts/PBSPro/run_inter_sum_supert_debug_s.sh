#!/bin/sh

# Run the interactive summarisation simulations with SUPERT

# Job name
#PBS -N intsums_deb

# Output file
#PBS -o pbs_intersumsupdebugs_output.log

# Error file
#PBS -e pbs_intersumsupdebugs_err.log

# request resources and set limits
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=24:mem=128GB:ompthreads=24
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/pytorch

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /work/es1595/text_ranking_bayesian_optimisation

#  run the script for each DUC dataset with GPPL-IMP, GPPL-UNPA, GPPL-EIG, GPPL-Random, BT-Random.
python -u stage1_active_pref_learning.py GPPLHHs 0 duc02_supert_imp_gpplhh_10 "[imp]" results_lstest5 24 DUC2004 10 supert 200 2 results_lstest5
python -u stage1_active_pref_learning.py GPPLHHs 0 duc02_supert_imp_gpplhh "[imp]" results_lstest5 24 DUC2004 100 supert 200 2 results_lstest5

# To submit: qsub run_bert_cqa.sh
# To display the queue: qstat -Q gpu (this is usually where the GPU job ends up)
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>

# To monitor job progress:
# qstat -f | grep exec_host
# Find the node where this job is running.
# ssh to the node.
# tail /var/spool/pbs/spool/<job ID>.bp1.OU
