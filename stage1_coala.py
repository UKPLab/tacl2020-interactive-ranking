import datetime
import json
import os
import pickle
import re
import sys
import pandas as pd
from joblib.parallel import Parallel, delayed

from resources import ROUGE_DIR, BASE_DIR
from summariser.oracle.lno_ref_values import SimulatedUser
from summariser.querier.expected_improvement_querier import ExpectedImprovementQuerier
from summariser.querier.expected_information_querier import InformationGainQuerier
from summariser.querier.gibbs_querier import GibbsQuerier
from summariser.querier.pairwise_uncertainty_querier import PairUncQuerier
from summariser.querier.pairwise_uncertainty_secondorder_querier import PairUncSOQuerier
from summariser.querier.thompson_querier import ThompsonTopTwoQuerier, ThompsonInformationGainQuerier
from summariser.querier.uncertainty_querier import UncQuerier
from summariser.utils.evaluator import evaluateReward, normaliseList
from summariser.querier.random_querier import RandomQuerier
from summariser.rouge.rouge import Rouge

import logging
logging.basicConfig(level=logging.DEBUG)

from summariser.querier.logistic_reward_learner import LogisticRewardLearner
from summariser.querier.GPPL_reward_learner import GPPLRewardLearner, GPPLHRewardLearner, GPPLHsRewardLearner
from random import seed
import numpy as np

res_dir = 'results_cqa'


def process_cmd_line_args(args):

    if len(args) > 1:
        learner_type_str = args[1]
    else:
        learner_type_str = 'LR'

    if len(args) > 2:
        n_debug = int(args[2])
    else:
        # to debug with a small subset of data, set this to a number of summaries
        n_debug = 0

    if len(args) > 3:
        output_folder_name_in = args[3]
    else:
        output_folder_name_in = -1

    if len(args) > 4:
        querier_types = args[4].strip('[]').split(',')
    else:
        querier_types = None

    if len(args) > 5 and args[5][0] != '-':
        root_dir = args[5]

        if not os.path.exists(root_dir + '/' + res_dir):
            os.mkdir(root_dir + '/' + res_dir)
        if not os.path.exists(root_dir + '/data'):
            os.mkdir(root_dir + '/data')
    else:
        root_dir = '.'

    if len(args) > 6:
        nthreads = int(args[6])
    else:
        nthreads = 0

    if len(args) > 7:
        n_inter_rounds = int(args[7])
    else:
        n_inter_rounds = n_debug if n_debug else 10

    if len(args) > 8:
        baseline = args[8] # this is the model used to obtain the initial predictions. It can be 'BERT' or 'COALA'.
        # Both load pre-computed values -- use a different script to produce them.
    else:
        baseline = 'COALA'

    if learner_type_str == 'LR':
        if querier_types is None:
            querier_types = ['random', 'unc']
        post_weight = 0.5  # trade off between the heuristic rewards and the pref-learnt rewards
        n_reps = 10

    elif learner_type_str == 'GPPL':
        if querier_types is None:
            querier_types = [
                'random',
                'pair_unc',
                'pair_unc_SO',
            ]  # ['ttt', 'tp', 'imp']# 'ttt' 'random' 'gibbs' 'unc' 'eig' 'tp' 'imp' 'eig'

        post_weight = 0.5
        n_reps = 5

    elif learner_type_str == 'GPPLH':
        if querier_types is None:
            querier_types = [
                'random',
                'pair_unc',
                'pair_unc_SO',
            ]
        post_weight = 1
        n_reps = 5

    elif learner_type_str == 'GPPLHH' or learner_type_str == 'GPPLHHtune' or learner_type_str == 'GPPLHHs':

        if querier_types is None:
            querier_types = [
                'random',
                'pair_unc',
                'pair_unc_SO',
                #'eig', # this should be very similar to pari_unc_SO so I think it's redundant
                'imp',
                # 'ttt', # this didn't work
                'tp'
            ]
        post_weight = 1

        print('Changing the number of repeats to 1 as there are no random initialisations with this method.')
        n_reps = 1

        if len(querier_types) == 1 and querier_types[0] == 'random':
            n_reps = 5

    elif learner_type_str == 'H':
        post_weight = 0
        querier_types = ['random']
        n_reps = 1

    first_rep = 0
    reps = range(first_rep, n_reps)

    seed(28923895)
    np.random.seed(1238549)
    # generate a list of seeds that can be used with all queriers in each repetition
    seeds = np.random.randint(1, 10000, n_reps)

    if learner_type_str == 'LR':
        learner_type = LogisticRewardLearner
    elif learner_type_str == 'GPPL':
        learner_type = GPPLRewardLearner
    elif learner_type_str == 'GPPLH' or learner_type_str == 'GPPLHH' \
            or learner_type_str == 'GPPLHHtune':
        # GPPL with heuristics as the prior mean
        learner_type = GPPLHRewardLearner
    elif learner_type_str == 'GPPLHHs':
        learner_type = GPPLHsRewardLearner
    else:
        learner_type = None

    return learner_type, learner_type_str, n_inter_rounds, output_folder_name_in, querier_types, root_dir, post_weight, \
           reps, seeds, n_debug, nthreads, baseline



def learn_model(question_id, ref_values, querier_type, learner_type, learner_type_str, summary_vectors, heuristics_list,
                post_weight, n_inter_rounds, all_result_dic, n_debug, output_path, n_threads, gold_idx, prior_scale=1.0,
                prior_offset=0.0, reload_rewards=False):

    print('\n---question no. {}---'.format(question_id))

    if learner_type is not None:
        learner_type_label = learner_type.__name__
    else:
        learner_type_label = 'nolearner'

    reward_file = output_path + '/rewards_%s_%s_%s.json' % (question_id, querier_type, learner_type_label)
    # if this has already been done, skip it!
    if os.path.exists(reward_file) and reload_rewards:

        # reload the pre-computed rewards
        with open(reward_file, 'r') as fh:
            learnt_rewards = json.load(fh)

            print('Computing metrics...')
            metrics_dic = evaluateReward(learnt_rewards, ref_values, top_answer=gold_idx)

            for metric in metrics_dic:
                print('metric {} : {}'.format(metric, metrics_dic[metric]))
                if metric in all_result_dic:
                    all_result_dic[metric].append(metrics_dic[metric])
                else:
                    all_result_dic[metric] = [metrics_dic[metric]]
            return learnt_rewards

    if n_debug:
        ref_values = ref_values[:n_debug]

    oracle = SimulatedUser(ref_values, 0.3)  # LNO-0.1

    if querier_type == 'gibbs':
        querier = GibbsQuerier(learner_type, summary_vectors, heuristics_list, post_weight)
    elif querier_type == 'unc':
        querier = UncQuerier(learner_type, summary_vectors, heuristics_list, post_weight)
    elif querier_type == 'pair_unc':
        querier = PairUncQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads,
                                 prior_scale=prior_scale, prior_offset=prior_offset)
    elif querier_type == 'pair_unc_SO':
        querier = PairUncSOQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads)
    elif querier_type == 'imp':
        querier = ExpectedImprovementQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads)
    elif querier_type == 'eig':
        querier = InformationGainQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads)
    elif querier_type == 'ttt':
        querier = ThompsonTopTwoQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads)
    elif querier_type == 'tp':
        querier = ThompsonInformationGainQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads)
    else:
        querier = RandomQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads)

    log = []

    if 'tune' in learner_type_str:
        querier.tune_learner()

    if learner_type_str == 'GPPLHH' or learner_type_str == 'GPPLHHtune' \
            or learner_type_str == 'GPPLHHs':
        # the first sample should not use the default of random selection, but should already apply
        # the chosen AL strategy
        querier.random_initial_sample = False

    if learner_type_str != 'H':
        for round in range(n_inter_rounds):
            sum1, sum2 = querier.getQuery(log)
            pref = oracle.getPref(sum1, sum2)
            log.append([[sum1, sum2], pref])
            if querier_type != 'random' or round == n_inter_rounds - 1:
                # with random querier, don't train until the last iteration as the intermediate results are not used
                querier.updateRanker(log)

    print('Active learning complete. Now getting mixed rewards')

    print('Computing metrics...')
    learnt_rewards = querier.getMixReward()
    metrics_dic = evaluateReward(learnt_rewards, ref_values, top_answer=gold_idx)

    for metric in metrics_dic:
        print('metric {} : {}'.format(metric, metrics_dic[metric]))
        if metric in all_result_dic:
            all_result_dic[metric].append(metrics_dic[metric])
        else:
            all_result_dic[metric] = [metrics_dic[metric]]

    print('Saving the rewards for this question...')

    with open(reward_file, 'w') as fh:
        json.dump(learnt_rewards, fh)

    return learnt_rewards

def save_result_dic(all_result_dic, output_path, rep, q_cnt, querier_type, learner_type_str, n_inter_rounds):
    # Compute and save metrics for this topic
    print('=== (rep={}) RESULTS UNTIL QUESTION {}, QUERIER {}, LEARNER {}, INTER ROUND {} ===\n'.format(
        rep, q_cnt, querier_type.upper(), learner_type_str, n_inter_rounds
    ))
    for metric in all_result_dic:
        print('{} : {}'.format(metric, np.mean(all_result_dic[metric])))

    with open(output_path + '/metrics_%s_%s_%i.json' % (querier_type, learner_type_str,
                                                        n_inter_rounds), 'w') as fh:
        json.dump(all_result_dic, fh)


def save_selected_results(output_path, all_result_dic, selected_means, selected_vars, selected_means_allreps,
                          selected_vars_allreps, chosen_metrics, method_names, this_method_idx):
    for m, metric in enumerate(chosen_metrics):

        selected_means[this_method_idx, m] = np.mean(all_result_dic[metric])
        selected_vars[this_method_idx, m] = np.var(all_result_dic[metric])

        selected_means_allreps[this_method_idx, m] += selected_means[this_method_idx, m]
        selected_vars_allreps[this_method_idx, m] += selected_vars[this_method_idx, m]

    df = pd.DataFrame(np.concatenate((np.array(method_names)[:, None], selected_means, selected_vars), axis=1),
                      columns=np.concatenate((['Method'], chosen_metrics,
                                              ['%s var' % metric for metric in chosen_metrics]))).set_index('Method')

    df.to_csv(output_path + '/table.csv')


def save_selected_results_allreps(output_path, selected_means_allreps, selected_vars_allreps, chosen_metrics, method_names, nreps):
    selected_means_allreps /= float(nreps)
    selected_vars_allreps /= float(nreps)

    df = pd.DataFrame(np.concatenate((np.array(method_names)[:, None], selected_means_allreps,
                                      selected_vars_allreps), axis=1),
                      columns=np.concatenate((['Query_type'], chosen_metrics,
                                              ['%s var' % metric for metric in chosen_metrics]))).set_index(
        'Query_type')

    filename = output_path + '/table_all_reps.csv'
    print('Saving summary of all results to %s' % filename)
    df.to_csv(filename)


def make_output_dir(root_dir, output_folder_name, rep):
    if output_folder_name == -1:
        output_folder_name = datetime.datetime.now().strftime('started-%Y-%m-%d-%H-%M-%S')
    else:
        output_folder_name = output_folder_name + '_rep%i' % rep

    if not os.path.exists(root_dir + '/' + res_dir):
        os.mkdir(root_dir + '/' + res_dir)

    output_path = root_dir + '/' + res_dir + '/%s' % output_folder_name
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    return output_path


def compute_ref_values(aidx, answer, gold_filename):
    answer = re.sub('<[^<]+>', "", answer)
    if np.mod(aidx, 20) == 0:
        print('computing ref value for answer: %i' % aidx)
    rouge_scorer = Rouge(ROUGE_DIR, BASE_DIR, True)
    R1, R2, RL, RSU = rouge_scorer(answer, [[gold_filename, None]], len(answer))
    rouge_scorer.clean()

    return RL


if __name__ == '__main__':

    '''
    Command line arguments:
    python stage1_coala.py reward_learner_type n_debug output_folder_name querier_types

    reward_learner_type -- can be LR, GPPL (mixes the posterior with the heuristics), GPPLH (uses heuristic as a prior), 
    GPPLHH (uses heuristic as a prior and a heuristic to select the initial sample). The best performer so far is GPPLHH.

    n_debug -- set to 0 if you are not debugging; set to a higher number to select a subsample of the data for faster 
    debugging of the main setup.

    output_folder_name -- this name will be used to store your results (metrics) and the rewards produced by the learner.
    This will be a subfolder of ./results_coala/ .

    querier_types -- a list of querier types. If the reward learner is LR, you can pass any subset of [random, unc].
    If the reward learner is any of the GPPL variants, you can pass [random, pair_unc, pair_unc_SO, tp, imp]. The best
    performers are tp and imp.    

    '''

    learner_type, learner_type_str, n_inter_rounds, output_folder_name, querier_types, root_dir, post_weight, reps, \
            seeds, n_debug, n_threads, baseline = process_cmd_line_args(sys.argv)

    max_qs = -1  # set to greater than zero to use a subset of topics for debugging
    folders = []

    nqueriers = len(querier_types)
    chosen_metrics = ['ndcg_at_1%', 'accuracy', 'pcc', 'ndcg_at_5%']  # , 'tau', 'ndcg_at_5%', 'ndcg_at_10%', 'rho']

    topics = ['cooking', 'apple', 'cooking', 'travel']

    for topic in topics:

        selected_means_allreps = np.zeros((nqueriers, len(chosen_metrics)))
        selected_vars_allreps = np.zeros((nqueriers, len(chosen_metrics)))

        # Loading the data for the baseline and gold labels:
        # qa_list, vec_list and pred_list each contain an entry for one question in the dataset.
        # Each entry of qa_list is a dict containing 'gold_answer' and a list of 'pooled_answers'.
        # Each entry of vec_list is a list of vector representations of the pooled answers + gold answer.
        # Each entry of pred_list is a list of baseline predictions for each of the pooled answers + gold_answer.
        if baseline == 'COALA':
            fname = 'data/cqa_base_models/coala_vec_pred/qa_vec_coala/se_%s_coala.qa_vec_pred' % topic
            qa_list, vec_list, pred_list = pickle.load(open(fname, 'rb'), encoding='latin1')
        elif baseline == 'BERT':
            fname_text = 'data/cqa_base_models/BERT_vec_pred/%s_text.tsv' % topic
            fname_numerical = 'data/cqa_base_models/BERT_vec_pred/%s_num.tsv' % topic
            # there is a separate csv file for each question

            # the tsv file contains a row per 'pooled' answer + a row at the end for the gold answer.
            # The columns are: 'answer', 'prediction', 'vector'.
            qdata = pd.read_csv(fname_text, '\t', header=0)
            answers = qdata['answer'].values
            qids = qdata['qid'].values
            isgold = qdata['isgold'].values

            vdata = pd.read_csv(fname_numerical, '\t', header=0).to_numpy(dtype=float)[:, 1:]
            preds = vdata[:, 0]
            print('Predictions: ')
            print(preds)

            vectors = vdata[:, 1:]

            qa_list = []
            vec_list = []
            pred_list = []

            uqids = np.unique(qids)
            for qidx, qid in enumerate(uqids):
                qidxs = qids == qid
                print('no. rows for question %i = %i' % (qidx, np.sum(qidxs)))
                qanswers = answers[qidxs]
                qgoldidx = np.argwhere(isgold[qidxs]).flatten()[0]
                qa_list.append({'gold_answer': qanswers[qgoldidx], 'pooled_answers': qanswers})

                qvec_list = vectors[qidxs]
                print('no. pooled answers for question %i = %i' % (qidx, len(qvec_list)))
                print(qvec_list.shape)

                goldvector = vectors[qgoldidx][None, :]
                print(goldvector.shape)

                qvec_list = np.concatenate((qvec_list, goldvector), axis=0)
                print(qvec_list.shape)

                qpred_list = preds[qidxs]
                qpred_list = np.append(qpred_list, preds[qgoldidx])

                vec_list.append(qvec_list)
                pred_list.append(qpred_list)

        print('sanity check')
        print('No. QA pairs = %i' % len(qa_list))
        print('No. vectors = %i' % np.array(vec_list).shape[0])
        print('No. prior predictions = %i' % len(pred_list))
        assert len(qa_list) == np.array(vec_list).shape[0] == len(pred_list)
        print('{} questions in total'.format(len(qa_list)))

        first_question = 0

        output_folder_topic = output_folder_name + '_%s' % topic

        for rep in reps:

            selected_means = np.zeros((nqueriers, len(chosen_metrics)))
            selected_vars = np.zeros((nqueriers, len(chosen_metrics)))

            output_path = make_output_dir(root_dir, output_folder_topic, rep)

            # saves a list of result folders containing repeats from the same run
            folders.append(output_path)
            with open(output_path + '/folders.txt', 'w') as fh:
                for folder_name in folders:
                    fh.write(folder_name + '\n')

            figs = []

            avg_figs = []

            for qidx, querier_type in enumerate(querier_types):

                seed(seeds[rep])
                np.random.seed(seeds[rep])

                ### store all results
                all_result_dic = {}
                q_cnt = 0

                amax = 0

                for question_id in range(first_question,  len(qa_list)):

                    print('\n=====(repeat {}) TOPIC {}, QUESTION {}, QUERIER {}, INTER ROUND {}====='.format(rep,
                            topic, question_id, querier_type.upper(), n_inter_rounds))

                    q_cnt += 1
                    if max_qs > 0 and q_cnt > max_qs or (n_debug and q_cnt > 1):
                        continue

                    # the last item in these lists corresponds to the gold answer, so cut it out
                    summary_vectors = vec_list[question_id][:-1]
                    heuristic_list = normaliseList(pred_list[question_id][:-1])

                    original_gold_answer = qa_list[question_id]['gold_answer']
                    pool_answers = qa_list[question_id]['pooled_answers']
                    # compute reference values as overlap with gold answer

                    gold_answer = re.sub('<[^<]+>', "", original_gold_answer)
                    # cache it to file
                    if not os.path.exists('data/cache'):
                        os.mkdir('data/cache')
                    gold_filename = 'data/cache/cqa_cache_%s_%i_%s.txt' % (topic, question_id, baseline)
                    if not os.path.exists(gold_filename):
                        with open(gold_filename, 'w') as fh:
                            fh.writelines(gold_answer)
                    # print('Computing reference scores against gold answer: %s' % gold_answer)

                    if n_debug:
                        heuristic_list = heuristic_list[:n_debug]
                        summary_vectors = summary_vectors[:n_debug]
                        pool_answers = pool_answers[:n_debug]

                    ref_filename = 'data/cqa_ref_scores/%s_ref_vals_rougel_lno03_%s_%i.txt' % \
                                   (baseline.lower(), topic, question_id)

                    if not os.path.exists(ref_filename):
                        ref_values = []

                        if len(pool_answers) == 1:
                            print('SKIPPING A QUESTION WITH BAD DATA')
                            print('Data dict contains: ')
                            print(qa_list[question_id].keys())
                            continue

                        ref_values = Parallel(n_jobs=20, backend='threading')(
                            delayed(compute_ref_values)(aidx, answer, gold_filename)
                            for aidx, answer in enumerate(pool_answers))

                        ref_values = normaliseList(ref_values)

                        with open(ref_filename, 'w') as fh:
                            json.dump(ref_values, fh)
                    else:
                        with open(ref_filename, 'r') as fh:
                            ref_values = json.load(fh)

                    print('Number of reference values for question %i = %i' % (question_id, len(ref_values)))

                    learn_model(question_id, ref_values, querier_type, learner_type, learner_type_str,
                                summary_vectors, heuristic_list, post_weight, n_inter_rounds, all_result_dic, n_debug,
                                output_path, n_threads, np.where(np.array(pool_answers) == original_gold_answer)[0][0])

                    save_result_dic(all_result_dic, output_path, rep, q_cnt, querier_type, learner_type_str,
                                    n_inter_rounds)

                save_selected_results(output_path, all_result_dic, selected_means, selected_vars, selected_means_allreps,
                                  selected_vars_allreps, chosen_metrics, querier_types, qidx)

        save_selected_results_allreps(output_path, selected_means_allreps, selected_vars_allreps, chosen_metrics,
                                      querier_types, len(reps))