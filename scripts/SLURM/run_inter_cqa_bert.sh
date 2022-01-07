#!/bin/sh

# Run the interactive cQA simulations using BERT-cQA model

#SBATCH -t 72:00:00
#SBATCH -J intcqa_bert_quick
#SBATCH --mail-user=edwin.simpson@bristol.ac.uk
#SBATCH --mail-type=FAIL
#SBATCH -e /user/work/es1595/intercqabert.err.%j
#SBATCH -o /user/work/es1595/intercqabert.out.%j
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH --exclusive

#  load required modules
module load lang/python/anaconda/pytorch

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /user/home/es1595/tacl2020-interactive-ranking
export OMP_NUM_THREADS=24

python -u stage1_coala.py GPPLHH 0 cqa_bert_imp_gpplhh_4 "[imp]" . 4 4 BERT
