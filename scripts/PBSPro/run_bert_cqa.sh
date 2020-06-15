#!/bin/sh

# Job name
#PBS -N bertcqa 

# Output file
#PBS -o pbs_cqa_output.log

# Error file
#PBS -e pbs_cqa_err.log

# request resources and set limits
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:ngpus=4:mem=32GB
#:ompthreads=24
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/pytorch lang/cuda

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /work/es1595/text_ranking_bayesian_optimisation

#  run the script
#python -u BERT_cQA.py apple
#python -u BERT_cQA.py cooking
python -u BERT_cQA.py travel

# To submit: qsub run_bert_cqa.sh
# To display the queue: qstat -Q gpu (this is usually where the GPU job ends up)
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>

# To monitor job progress:
# qstat -f | grep exec_host
# Find the node where this job is running.
# ssh to the node.
# tail /var/spool/pbs/spool/<job ID>.bp1.OU
