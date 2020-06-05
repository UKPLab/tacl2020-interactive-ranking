import json
import os
import pickle
import re
import numpy as np
import pandas as pd

from summariser.utils.misc import normaliseList

imp = 'results_coala/lno03_gpplhh_apple_rep0/'

topic = 'apple'

fname = 'coala_vec_pred/qa_vec_coala/se_%s_coala.qa_vec_pred' % topic
qa_list, vec_list, pred_list = pickle.load(open(fname, 'rb'), encoding='latin1')

max_no_egs = 5 # find five examples
no_egs_correct = 0
no_egs_wrong = 0

for question_id in range(len(qa_list)):

    if no_egs_correct == max_no_egs and no_egs_wrong == max_no_egs:
        break

    # the last item in these lists corresponds to the gold answer, so cut it out
    summary_vectors = vec_list[question_id][:-1]
    heuristic_list = normaliseList(pred_list[question_id][:-1])

    original_gold_answer = qa_list[question_id]['gold_answer']
    pool_answers = qa_list[question_id]['pooled_answers']
    # compute reference values as overlap with gold answer

    gold_answer = re.sub('<[^<]+>', "", original_gold_answer)
    # cache it to file
    gold_filename = 'data/coala_cache_%s_%i.txt' % (topic, question_id)
    if not os.path.exists(gold_filename):
        with open(gold_filename, 'w') as fh:
            fh.writelines(gold_answer)

    gold_idx = np.where(np.array(pool_answers) == original_gold_answer)[0][0]

    # compute whether heuristic is correct
    coala_prediction = np.argmax(heuristic_list)

    # load my rewards
    reward_file = imp + '/rewards_%s_imp_GPPLHRewardLearner.json' % (question_id)

    with open(reward_file, 'r') as fh:
        rewards = json.load(fh)

    imp_prediction = np.argmax(rewards)

    # compute whether imp is correct
    if (imp_prediction == gold_idx) and (coala_prediction != gold_idx) and no_egs_correct < max_no_egs:
        print('IMP was correct, COALA was wrong: ')
        print('Question: %s' % qa_list[question_id]['question'])
        print('%s & ' % gold_answer)
        print('%s & ' % pool_answers[coala_prediction])
        no_egs_correct += 1

    if (imp_prediction != gold_idx) and (coala_prediction != gold_idx) and no_egs_wrong < max_no_egs:
        print('Both were wrong: ')
        print('Question: %s' % qa_list[question_id]['question'])
        print('%s &' % gold_answer)
        print('%s &' % pool_answers[coala_prediction])
        print('%s &' % pool_answers[imp_prediction])
        print('%-------')

        no_egs_wrong += 1

