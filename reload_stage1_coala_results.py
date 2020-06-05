import json
import os
import pickle
import sys
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from stage1_coala import process_cmd_line_args, save_selected_results, save_selected_results_allreps

if __name__ == '__main__':

    learner_type, learner_type_str, n_inter_rounds, output_folder_name, querier_types, root_dir, post_weight, reps, \
    seeds, n_debug, n_threads = process_cmd_line_args(sys.argv)

    topics = ['cooking', 'apple', 'travel']

    for topic in topics:

        # get the output path of the last repetition of the experiment
        output_path = None
        rep_i = 0
        while output_path is None or not os.path.exists(output_path):
            rep_i -= 1
            if np.isscalar(reps):
                rep = reps
            else:
                print('loading repeat idx %i in %s' % (rep_i, str(reps)))
                rep = reps[rep_i]

            if rep < 0:
                print('Could not find any results :(')
                sys.exit(0)

            output_folder_r = output_folder_name + '_%s_rep%i' % (topic, rep)
            output_path = root_dir + '/results_coala/%s' % output_folder_r
            print('Checking %s' % output_path)

        # get a list of folders relating to this experiment
        nqueriers = len(querier_types)

        chosen_metrics = ['ndcg_at_1%', 'accuracy', 'pcc', 'ndcg_at_5%', 'ndcg_at_10%']

        nreps = rep + 1 # number of repeats that were actually completed

        selected_means_allreps = np.zeros((nqueriers, len(chosen_metrics)))
        selected_vars_allreps = np.zeros((nqueriers, len(chosen_metrics)))

        for r in range(nreps):
            output_folder_r = output_folder_name + '_%s_rep%i' % (topic, r)
            output_path = root_dir + '/results_coala/%s' % output_folder_r

            selected_means = np.zeros((nqueriers, len(chosen_metrics)))
            selected_vars = np.zeros((nqueriers, len(chosen_metrics)))

            for qidx, querier_type in enumerate(querier_types):

                filename = '%s/metrics_%s_%s_%i.json' % (output_path, querier_type, learner_type_str, n_inter_rounds)

                if not os.path.exists(filename):
                    print('Cannot find the results file %s' % filename)
                    continue
                else:
                    print('Found the results file %s' % filename)

                with open(filename, 'r') as fh:
                    all_result_dic = json.load(fh)

                accs = []
                fname = 'coala_vec_pred/qa_vec_coala/se_%s_coala.qa_vec_pred' % topic

                if learner_type is not None:
                    learner_type_label = learner_type.__name__
                else:
                    learner_type_label = 'nolearner'

                qa_list, vec_list, pred_list = pickle.load(open(fname, 'rb'), encoding='latin1')
                for question_id in range(len(qa_list)):
                    reward_file = output_path + '/rewards_%s_%s_%s.json' % (
                    question_id, querier_type, learner_type_label)

                    if not os.path.exists(reward_file):
                        continue

                    with open(reward_file, 'r') as fh:
                        learnt_values = json.load(fh)

                    # recompute the accuracies
                    gold_answer = qa_list[question_id]['gold_answer']
                    pool_answers = qa_list[question_id]['pooled_answers']

                    gold_idx = np.where(np.array(pool_answers) == gold_answer)[0][0]
                    accs.append(100 * float(gold_idx == np.argmax(learnt_values)))

                all_result_dic['accuracy'] = accs

                save_selected_results(output_path, all_result_dic, selected_means, selected_vars, selected_means_allreps,
                                  selected_vars_allreps, chosen_metrics, querier_types, qidx)

        print('Saving result summary to %s' % output_path)
        save_selected_results_allreps(output_path, selected_means_allreps, selected_vars_allreps, chosen_metrics,
                                          querier_types, nreps)
