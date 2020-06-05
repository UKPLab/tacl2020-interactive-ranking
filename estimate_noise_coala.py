import datetime
import json
import os
import pickle
import re
import sys
import pandas as pd
from sklearn.metrics import accuracy_score

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

# possible future work: plot the results at each AL iteration (or at 10, 50, 100) to see if we really need so many when the prior is given.
#TODO: reduce the number of interactions to 10 to make the experiments quicker -- how does it affect results? --> so far
# not good. So try the next step...
#TODO: test with perfect oracle

if __name__ == '__main__':

    '''
    Estimate the accuracy of the pairwise labels from the simulated user at a given 'temperature' level.
    '''

    if len(sys.argv) > 1:
        temp = float(sys.argv[1])
    else:
        temp = 2.5

    max_qs = -1  # set to greater than zero to use a subset of topics for debugging
    folders = []

    topic = 'apple'

    fname = 'coala_vec_pred/qa_vec_coala/se_%s_coala.qa_vec_pred' % topic
    qa_list, vec_list, pred_list = pickle.load(open(fname, 'rb'), encoding='latin1')

    print('sanity check')
    assert len(qa_list) == len(vec_list) == len(pred_list)
    print('{} questions in total'.format(len(qa_list)))

    first_question = 0
    n_inter_rounds = 1000

    user_log = []
    true_log = []

    for question_id in range(100):

        print('\n===== TOPIC {}, QUESTION {}, INTER ROUND {}====='.format(topic, question_id, n_inter_rounds))

        # the last item in these lists corresponds to the gold answer, so cut it out
        summary_vectors = vec_list[question_id][:-1]
        heuristic_list = pred_list[question_id][:-1]

        # the last item in these lists corresponds to the gold answer, so cut it out
        original_gold_answer = qa_list[question_id]['gold_answer']
        pool_answers = qa_list[question_id]['pooled_answers']
        # compute reference values as overlap with gold answer

        gold_answer = re.sub('<[^<]+>', "", original_gold_answer)
        # cache it to file
        gold_filename = 'data/coala_cache_%s_%i.txt' % (topic, question_id)
        if not os.path.exists(gold_filename):
            with open(gold_filename, 'w') as fh:
                fh.writelines(gold_answer)
        print('Computing reference scores against gold answer: %s' % gold_answer)

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
                if np.mod(aidx, 20) == 0:
                    print('computing ref value for answer: %i' % aidx)

                rouge_scorer = Rouge(ROUGE_DIR, BASE_DIR, True)
                R1, R2, RL, RSU = rouge_scorer(answer, [[gold_filename, None]], len(answer))
                rouge_scorer.clean()

                # this is the combination learned for summarisation
                # ref_values.append(R1 / 0.48 + R2 / 0.212 + RSU / 0.195)

                #it's more common to use ROUGE L for QA
                ref_values.append(RL)

            ref_values = normaliseList(ref_values)

            with open(ref_filename, 'w') as fh:
                json.dump(ref_values, fh)
        else:
            with open(ref_filename, 'r') as fh:
                ref_values = json.load(fh)


        # sample some pairs
        log = []
        querier = RandomQuerier(LogisticRewardLearner, summary_vectors, heuristic_list, 0.5, 1)
        oracle = SimulatedUser(ref_values, temp)

        for round in range(n_inter_rounds):
            sum1, sum2 = querier.getQuery(log)
            pref = oracle.getPref(sum1, sum2)

            # get the user's label
            user_log.append(pref)

            # get the 'true' labels according to the ordering of the ref_values
            true_log.append(ref_values[sum1] < ref_values[sum2])

    # compute the accuracy
    acc = accuracy_score(true_log, user_log)

    print('The simulated user has accuracy of %f' % acc)