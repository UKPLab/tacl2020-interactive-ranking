#!/bin/sh

# Run the interactive summarisation simulations with SUPERT

# Job name
#PBS -N intsumreaSUM

# Output file
#PBS -o pbs_intersumreaSUM04_output.log

# Error file
#PBS -e pbs_intersumreaSUM04_err.log

# request resources and set limits
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=24:mem=128GB:ompthreads=24
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/pytorch

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /work/es1595/text_ranking_bayesian_optimisation

# Run the script using heuristics only and no interactions.
python -u stage1_active_pref_learning.py GPPL 0 duc04_reaper_gppl_20 "[imp,random,pair_unc]" . 4 DUC2004 20 april 200 1 results_reaper_1 1

# To submit: qsub run_bert_cqa.sh
# To display the queue: qstat -Q gpu (this is usually where the GPU job ends up)
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>

# To monitor job progress:
# qstat -f | grep exec_host
# Find the node where this job is running.
# ssh to the node.
# tail /var/spool/pbs/spool/<job ID>.bp1.OU
