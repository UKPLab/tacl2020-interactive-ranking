import datetime
import json
import os
import pickle
import re
import sys
import pandas as pd
from joblib import Parallel, delayed

from resources import ROUGE_DIR, BASE_DIR
from stage1_coala import learn_model, make_output_dir, compute_ref_values
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

if __name__ == '__main__':

    '''
    Run the stage1 coala tests with different posterior weights and scalings for the GPPL prior to find the best
    settings. The optimisation uses grid search to maximise accuracy on a subset of the apple questions (first 250).

    '''
    n_threads = 2
    output_folder_name = 'coala_weight_tuning_rougel'
    root_dir = '.'

    topic = 'apple'

    fname = 'data/coala_vec_pred/qa_vec_coala/se_%s_coala.qa_vec_pred' % topic
    qa_list, vec_list, pred_list = pickle.load(open(fname, 'rb'), encoding='latin1')

    print('sanity check')
    assert len(qa_list) == len(vec_list) == len(pred_list)
    print('{} questions in total'.format(len(qa_list)))

    output_folder_topic = output_folder_name + '_%s' % topic

    rep = 1000  # just use this so we don't clash with other runs

    dev_sample_size = 250

    output_path = make_output_dir(root_dir, output_folder_topic, rep)

    seed(28923895)
    np.random.seed(1238549)
    # generate a list of seeds that can be used with all queriers in each repetition
    seed = np.random.randint(1, 10000, 1)

    querier_type = 'unc'
    n_inter_rounds = 10
    learner_type_str = 'LR'
    learner_type = LogisticRewardLearner

    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    weight_scores = np.zeros(len(weights))

    for w, weight in enumerate(weights):

        all_result_dic = {}

        q_cnt = 0
        for question_id in range(dev_sample_size):

            print('\n=====(repeat {}) TOPIC {}, QUESTION {}, QUERIER {}, INTER ROUND {}====='.format(rep,
                        topic, question_id, querier_type.upper(), n_inter_rounds))

            q_cnt += 1

            # the last item in these lists corresponds to the gold answer, so cut it out
            summary_vectors = vec_list[question_id][:-1]
            heuristic_list = pred_list[question_id][:-1]

            original_gold_answer = qa_list[question_id]['gold_answer']
            pool_answers = qa_list[question_id]['pooled_answers']
            # compute reference values as overlap with gold answer

            gold_answer = re.sub('<[^<]+>', "", original_gold_answer)
            # cache it to file
            gold_filename = 'data/cache/coala_cache_%s_%i.txt' % (topic, question_id)
            if not os.path.exists(gold_filename):
                with open(gold_filename, 'w') as fh:
                    fh.writelines(gold_answer)
            print('Computing reference scores against gold answer: %s' % gold_answer)

            # ref_filename = 'data/coala_ref_vals_%s_%i.txt' % (topic, question_id)
            ref_filename = 'data/coala_ref_vals_rougel_%s_%i.txt' % (topic, question_id)
            if not os.path.exists(ref_filename):
                ref_values = []

                if len(pool_answers) == 1:
                    print('SKIPPING A QUESTION WITH BAD DATA')
                    print('Data dict contains: ')
                    print(qa_list[question_id].keys())
                    continue

                #for aidx, answer in enumerate(pool_answers):
                    # RL = compute_ref_values()
                    #
                    #it's more common to use ROUGE L for QA
                 #   ref_values.append(RL)

                ref_values = Parallel(n_jobs=12, backend='threading')(delayed(compute_ref_values)(aidx, answer, gold_filename)
                                                         for aidx, answer in enumerate(pool_answers))

                ref_values = normaliseList(ref_values)

                with open(ref_filename, 'w') as fh:
                    json.dump(ref_values, fh)
            else:
                with open(ref_filename, 'r') as fh:
                    ref_values = json.load(fh)

            learn_model(question_id, ref_values, querier_type, learner_type, learner_type_str,
                summary_vectors, heuristic_list, weight, n_inter_rounds, all_result_dic, 0,
                output_path, n_threads, np.where(np.array(pool_answers) == original_gold_answer)[0][0],
                reload_rewards=False)

        weight_scores[w] = np.mean(all_result_dic['accuracy']) #'ndcg_at_1%'])

    print('BEST WEIGHT IS %f with accuracy %f' % (weights[np.argmax(weight_scores)], np.max(weight_scores)) )
    np.savetxt(output_path + '/weight_scores.txt', np.concatenate((weights, weight_scores)))

    querier_type = 'pair_unc'
    n_inter_rounds = 10
    learner_type_str = 'GPPLHH'
    learner_type = GPPLHRewardLearner

    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    offsets = [-0.5, -0.1, 0, 0.1, 0.5]
    scale_offset_scores = np.zeros((len(scales), len(offsets)))

    for s, scale in enumerate(scales):
        for o, offset in enumerate(offsets):

            all_result_dic = {}

            q_cnt = 0
            for question_id in range(dev_sample_size):

                print('\n=====(repeat {}) TOPIC {}, QUESTION {}, QUERIER {}, INTER ROUND {}====='.format(rep,
                                                                                                         topic, question_id,
                                                                                                         querier_type.upper(),
                                                                                                         n_inter_rounds))

                q_cnt += 1

                # the last item in these lists corresponds to the gold answer, so cut it out
                summary_vectors = vec_list[question_id][:-1]
                heuristic_list = pred_list[question_id][:-1]

                original_gold_answer = qa_list[question_id]['gold_answer']
                pool_answers = qa_list[question_id]['pooled_answers']
                # compute reference values as overlap with gold answer

                gold_answer = re.sub('<[^<]+>', "", original_gold_answer)
                # cache it to file
                gold_filename = 'data/cache/coala_cache_%s_%i.txt' % (topic, question_id)
                if not os.path.exists(gold_filename):
                    with open(gold_filename, 'w') as fh:
                        fh.writelines(gold_answer)
                print('Computing reference scores against gold answer: %s' % gold_answer)

                # ref_filename = 'data/coala_ref_vals_%s_%i.txt' % (topic, question_id)
                ref_filename = 'data/coala_ref_vals_rougel_%s_%i.txt' % (topic, question_id)
                if not os.path.exists(ref_filename):
                    ref_values = []

                    if len(pool_answers) == 1:
                        print('SKIPPING A QUESTION WITH BAD DATA')
                        print('Data dict contains: ')
                        print(qa_list[question_id].keys())
                        continue

                    for aidx, answer in enumerate(pool_answers):
                        answer = re.sub('<[^<]+>', "", answer)
                        print('computing ref value for answer: %i' % aidx)
                        rouge_scorer = Rouge(ROUGE_DIR, BASE_DIR, True)
                        R1, R2, RL, RSU = rouge_scorer(answer, [[gold_filename, None]], len(answer))
                        rouge_scorer.clean()

                        # it's more common to use ROUGE L for QA
                        ref_values.append(RL)

                    ref_values = normaliseList(ref_values)

                    with open(ref_filename, 'w') as fh:
                        json.dump(ref_values, fh)
                else:
                    with open(ref_filename, 'r') as fh:
                        ref_values = json.load(fh)

                learn_model(question_id, ref_values, querier_type, learner_type, learner_type_str,
                            summary_vectors, heuristic_list, 1.0, n_inter_rounds, all_result_dic, 0,
                            output_path, n_threads, np.where(np.array(pool_answers) == original_gold_answer)[0][0],
                            reload_rewards=False)

            scale_offset_scores[s, o] = np.mean(all_result_dic['accuracy'])


    indexes = np.unravel_index(np.argmax(scale_offset_scores), dims=scale_offset_scores.shape)
    sidx = indexes[0]
    oidx = indexes[1]
    print('BEST SCALE IS %f AND OFFSET %f with accuracy %f' % (scales[sidx], offsets[oidx], np.max(scale_offset_scores)))
    np.savetxt(output_path + '/scale_offset_scores.txt', np.concatenate((np.array(scales)[:, None],
                                                                        scale_offset_scores), axis=1))