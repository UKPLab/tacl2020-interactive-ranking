# Interactive Text Ranking with Bayesian Optimisation: A Case Study on Community QA and Summarisation

## Contents

* [Introduction](#introduction)
* [Publications](#publications)
* [Project Structure](#project-structure)
* [Requirements and Installation](#requirements-and-installation)
* [Running the experiments](#running-the-experiments)

## Introduction

This package contains the experimental code for the TACL paper
'Interactive Text Ranking with Bayesian Optimisation: A Case Study on Community QA and Summarisation'.
This code implements the proposed Bayesian optimisation methods using GPPL.
The instructions are described below.

**Contact Person**

Edwin Simpson: edwin.simpson@bristol.ac.uk

Please feel free to contact us with any queries about running or understanding our code.

## Publication

Please cite the paper as follows:
```bibtex
@article{simpson2020interactive,
    title = "Interactive Text Ranking with {Bayesian} Optimisation: A Case Study on Community {QA} and Summarisation",
    author = "Simpson, Edwin and Gao, Yang and Gurevych, Iryna",
    journal = "Transactions of the Association of Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    pages = "to appear",
}
```

>**Abstract**
>For many NLP applications, 
>such as question answering and summarisation, 
>the goal is to select the best solution from a large space of candidates 
>to meet a particular user’s needs. To address the lack of user 
>or task-specific training data, we propose an interactive 
>text ranking approach that actively selects pairs of candidates, 
>from which the user selects the best. Unlike previous strategies, 
>which attempt to learn a ranking across the whole can- didate space, 
>our method employs Bayesian optimisation to focus the user’s labelling 
>effort on high quality candidates and integrate prior knowledge 
>to cope better with small data scenarios. We apply our method 
>to community question answering (cQA) and 
>extractive multi-document summarisation, find- ing that it significantly 
>outperforms existing interactive approaches. We also show that 
>the ranking function learned by our method is an effective 
>reward function for reinforcement learning, which improves 
>the state of the art for interactive summarisation.

## Project Structure

* . - the main scripts for running experiments are stored at the top level of this repository.
* data - contains the preprocessed CQA and summarisation datasets and the predictions of COALA on the test sets.
   * summariser - Implementation of active learning and user simulation with the following subfolders.
* summariser/oracle - logistic noisy oracle implementation.
* summariser/querier - implementations of the active learning acquisition functions, including the Bayesian optimisation methods.
* summariser/rl - reinforcement learner implementation.
* summariser/rouge - ROUGE implementation by Chin-Yew Lin.
* summariser/utils - utilities for reading and writing data, etc.
* summariser/vector - helper code for vector representations of summaries.

* flattened_rewards_plot - script for generating Figure 7
* gppl - the GPPL implementation; for more details, please see https://github.com/ukplab/tacl2018-preference-convincing.
* ref_free_metrics - SUPERT implementation; please see https://github.com/yg211/acl20-ref-free-eval.
* sentence_transformers - sentence Transformers implementation used by SUPERT; please see https://github.com/yg211/acl20-ref-free-eval.
* scripts - batch scripts for running experiments on PBSPro and SLURM clusters.

## Requirements and Installation

See the environment.yml file for a list of dependencies. You can use this file 
with conda to set up an environment with all the dependencies:

`
conda env create -f environment.yml
`

## Running the Experiments

### cQA experiments with COALA priors

The COALA priors have already been run, so we can
just run the interactive simulation with GPPL:
```
python stage1_coala.py GPPLHH 0 <output_dir_name> "[random,pair_unc,eig,imp,tp]" . 4 <num_interactions>
```

To run the experiments with BT:
```
python stage1_coala.py LR 0 <output_dir_name> "[random,unc]" . 4 <num_interactions>
```

### cQA experiments with BERT-cQA priors

First, we need to run BERT-cQA on each of the StackExchange topics:

```
python BERT_cQA.py apple
python BERT_cQA.py cooking
python BERT_cQA.py travel
```

To run the experiments with GPPL:
```
python stage1_coala.py GPPLHH 0 <output_dir_name> "[random,pair_unc,eig,imp,tp]" . 4 <num_interactions> BERT
```

To run the experiments with BT:
```
python stage1_coala.py LR 0 <output_dir_name> "[random,unc]" . 4 <num_interactions> BERT
```

### Summary ranking with REAPER priors

The experiment learns a ranker by querying the user for pairwise 
preferences over sampled candidate summaries.
 The candidates are stored in `./data/sampled_summaries` and were
already generated using `stage0_sample_summaries.py` (no need to rerun this script).

To run the interactive learning simulation with GPPL:

```
python stage1_active_pref_learning.py GPPLHH 0 <output_folder_name> "[random,pair_unc,eig,tp,imp]" . 4 <dataset> <num_interactions> april
```

`dataset` can be duc2001, duc2002, or duc2004.
 
To run with the BT setups:

```
python stage1_active_pref_learning.py LR 0 <output_folder_name> "[random,unc]" . 4 <dataset> <num_interactions> april
```

### Summary ranking with SUPERT priors

First, obtain the SUPERT prior predictions for the candidate summaries:
`python obtain_supert_scores.py`.

Then run the interactive learning simulation with GPPL with
SUPERT priors and bigram+ features:

```
python stage1_active_pref_learning.py GPPLHH 0 <output_folder_name> "[random,pair_unc,eig,tp,imp]" . 4 <dataset> <num_interactions> supertbigram+
```

Run the interactive learning simulation with GPPL with
SUPERT priors and SUPERT embeddings:

```
python stage1_active_pref_learning.py GPPLHH 0 <output_folder_name> "[random,pair_unc,eig,tp,imp]" . 4 <dataset> <num_interactions> supert
```

`dataset` can be duc2001, duc2002, or duc2004.
 
To run with the BT setups with bigram+:

```
python stage1_active_pref_learning.py LR 0 <output_folder_name> "[random,unc]" . 4 <dataset> <num_interactions> supertbigram+
```

BT setups with SUPERT embeddings:
```
python stage1_active_pref_learning.py LR 0 <output_folder_name> "[random,unc]" . 4 <dataset> <num_interactions> supert
```

### Progress Plots (Figures 3-6)

In `stage1_plot_progress.py`, set line 15 to choose which task to plot:
```
tasks = ['duc2001']  # ['supert_duc2001', 'supert_bi_duc2001']  # 'bertcqa'  # 'coala' #  #
```
See lines 79 to 113 to see which numbers of interactions
are required for the chosen task. 
Run the corresponding experiments as above, setting
the ouput directories to fit the expected pattern in 
 lines 79 to 113 of `stage1_plot_progress.py`.
Then run the script:
```
python stage1_plot_progress.py
```
The plots will be saved to './plots/progress_<task_name>'.

### Reinforcement learning for summary generation

If not already installed, run `cpan XML::DOM` to install Perl requirements.

Install WordNet:
```
cd summariser/rouge/ROUGE-RELEASE-1.5.5/data/
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
```

Run the reinforcement learner as follows:
```
python stage2_reinf_learning.py learnt GPPLHH 0 <output_folder_name> [imp]
```
The `output_folder_name` should match the location used in the previous task. This is where the rewards learned during 
the interactive simulation will have been stored.