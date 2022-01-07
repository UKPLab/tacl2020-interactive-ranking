#!/bin/sh

# Run the interactive cQA simulations using BERT-cQA model

#SBATCH -t 72:00:00
#SBATCH -J intcqa_bert_quick
#SBATCH --mail-user=edwin.simpson@bristol.ac.uk
#SBATCH --mail-type=FAIL
#SBATCH -e /work/scratch/es14puve/intercqabert.err.%j
#SBATCH -o /work/scratch/es14puve/intercqabert.out.%j
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=8192
#SBATCH --exclusive
#SBATCH -C avx

# request resources and set limits

#PBS -l select=1:ncpus=24:mem=128GB:ompthreads=24
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/pytorch

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /work/es1595/text_ranking_bayesian_optimisation
export OMP_NUM_THREADS=24

python -u stage1_coala.py GPPLHH 0 cqa_bert_DR2_imp_gpplhh_5 "[imp]" . 4 4 BERT
