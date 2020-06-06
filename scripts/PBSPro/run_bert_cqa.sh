#!/bin/sh

#PBS -N bertcqa // job name
#PBS -o pbs_output.log // output file
#PBS -e pbs_err.log // error file

# request resources and set limits
#PBS -l walltime==24:00:00
#PBS -l select=1:ncpus=24:ngpus=2:mem=32GB:ompthreads=24  // 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/pytorch lang/cuda

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /work/es1595/text_ranking_bayesian_optimisation

#  run the script
python -u BERT_cQA.py

# To submit: qsub run_bert_cqa.sh
# To display the queue: qstat -Q
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>
