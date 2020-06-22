import json
import os
import sys
from random import seed
import numpy as np

from stage1_active_pref_learning import learn_model, load_summary_vectors, save_selected_results, \
    save_result_dic, save_selected_results_allreps, make_output_dir, process_cmd_line_args
from resources import PROCESSED_PATH
from summariser.utils.corpus_reader import CorpusReader
from summariser.utils.reader import readSampleSummaries
from summariser.vector.vector_generator import Vectoriser
from summariser.rl.deep_td import DeepTDAgent
from summariser.utils.evaluator import evaluateSummary

# TODO add documentation on running cpan XML::DOM to install Perl requirements
# Same same for the wordnet database -- execture the following steps:
# cd summariser/rouge/ROUGE-RELEASE-1.5.5/data/
# rm WordNet-2.0.exc.db
# ./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db

def add_result(all_dic, result):
    for metric in result:
        if metric in all_dic:
            if isinstance(result[metric],list): all_dic[metric].extend(result[metric])
            else: all_dic[metric].append(result[metric])
        else:
            if isinstance(result[metric],list): all_dic[metric] = result[metric]
            else: all_dic[metric] = [result[metric]]

if __name__ == '__main__':

    '''
    Command line arguments:
    python stage2_reinf_learning.py reward_type reward_learner_type n_debug output_folder_name querier_types
    
    Suggested example:
    
    python stage2_reinf_learning.py learnt GPPLHH 0 test_gpplhh_imp [imp]

    reward_type -- can be either 'rouge', 'heuristic' or 'learnt'. If you select the latter, then the script will look
    for rewards produced by a learner using the specified querier. If these are not found, it will rerun the learning 
    process with the specified querier (i.e. it will repeat stage1 if its results are not found). 

    reward_learner_type -- can be LR, GPPL (mixes the posterior with the heuristics), GPPLH (uses heuristic as a prior), 
    GPPLHH (uses heuristic as a prior and a heuristic to select the initial sample). The best performer so far is GPPLHH.

    n_debug -- set to 0 if you are not debugging; set to a higher number to select a subsample of the data for faster 
    debugging of the main setup.

    output_folder_name -- this name will be used to store your results (metrics) and the rewards produced by the learner.
    This will be a subfolder of ./results/ .

    querier_types -- a list of querier types. If the reward learner is LR, you can pass any subset of [random, unc].
    If the reward learner is any of the GPPL variants, you can pass [random, pair_unc, pair_unc_SO, tig, imp]. The best
    performers are tig and imp.    

    '''

    if len(sys.argv) > 1:
        reward_types = sys.argv[1].strip('[]').split(',')
    else:
        reward_types = ['rouge']

    learner_type, learner_type_str, n_inter_rounds, output_folder_name, querier_types, root_dir, post_weight, reps, \
    seeds, n_debug, n_threads, dataset, feature_type, _, _, _ = process_cmd_line_args(sys.argv[1:])

    if 'learnt' in reward_types:
        for q in querier_types:
            reward_types.append('learnt_' + q)
        reward_types.remove('learnt')

    #parameters
    rl_episode = 2000
    strict = 5.0

    if dataset is None:
        dataset = 'DUC2001' # 'DUC2004'

    # read documents and ref. summaries
    reader = CorpusReader(PROCESSED_PATH)
    data = reader.get_data(dataset)

    chosen_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-SU4', 'WeightedSum']

    max_topics = -1  # set to greater than zero to use a subset of topics for debugging

    nqueriers = len(querier_types)
    selected_means_allreps = np.zeros((nqueriers, len(chosen_metrics)))
    selected_vars_allreps = np.zeros((nqueriers, len(chosen_metrics)))

    for rep in reps:

        selected_means = np.zeros((nqueriers, len(chosen_metrics)))
        selected_vars = np.zeros((nqueriers, len(chosen_metrics)))

        output_path = make_output_dir(root_dir, output_folder_name, rep)

        for ridx, reward_querier_type in enumerate(reward_types):

            seed(seeds[rep])
            np.random.seed(seeds[rep])

            if '_' in reward_querier_type:
                querier_type = reward_querier_type.split('_')[1]
                reward_type = reward_querier_type.split('_')[0]

            topic_cnt = 0
            all_result_dic = {}
            stage1_all_result_dic = {} # may be needed if we have to run stage 1 again

            for topic,docs,models in data:

                topic_cnt += 1
                if max_topics > 0 and topic_cnt > max_topics or (n_debug and topic_cnt > 1):
                    continue

                summaries, ref_values_dic, heuristic_list = readSampleSummaries(dataset, topic)
                vec = Vectoriser(docs)
                #summary_vectors = vec.getSummaryVectors(summaries)
                rl_agent = DeepTDAgent(vec,summaries,rl_episode,strict)

                result_dic = {}
                for model in models:

                    if reward_type == 'rouge':
                        model_name = model[0].split('/')[-1].strip()
                        print('\n---ref {}---'.format(model_name))
                        summary = rl_agent(ref_values_dic[model_name])

                    elif reward_type == 'heuristic':
                        summary = rl_agent(heuristic_list)

                    else: # learned reward function
                        model_name = model[0].split('/')[-1].strip()
                        print('\n---ref {}---'.format(model_name))

                        reward_file = output_path + '/rewards_%s_%s_%s_%s.json' % (topic, model_name,
                                                                                querier_type, learner_type.__name__)
                        if os.path.exists(reward_file):
                            # reload the pre-computed rewards
                            with open(reward_file, 'r') as fh:
                                learnt_rewards = json.load(fh)
                        else:
                            # the rewards we need don't exist, so learn them now
                            print('no learnt reward available. start learnign now')
                            summary_vectors = load_summary_vectors(summaries, dataset, topic, root_dir, docs,
                                                                   feature_type)

                            if n_debug:
                                heuristic_list = heuristic_list[:n_debug]
                                summary_vectors = summary_vectors[:n_debug]

                            learnt_rewards = learn_model(topic, model, ref_values_dic, querier_type, learner_type,
                                             learner_type_str, summary_vectors, heuristic_list, post_weight,
                                             n_inter_rounds, stage1_all_result_dic, n_debug, output_path, n_threads)

                        summary = rl_agent(learnt_rewards)

                    result = evaluateSummary(summary, model)
                    add_result(result_dic, result)
                    for metric in result:
                        print('metric {} : {}'.format(metric, result[metric]))

                add_result(all_result_dic, result_dic)
                print('\n=== RESULTS UNTIL TOPIC {}, EPISODE {}, STRICT {}, EPISODE {} ===\n'.format(topic_cnt,
                                                                                        rl_episode,strict,rl_episode))
                for metric in all_result_dic:
                    print('{} : {}'.format(metric, np.mean(all_result_dic[metric])))

                save_result_dic(all_result_dic, output_path, rep, topic_cnt, querier_type, learner_type_str,
                                n_inter_rounds)
            # save the results
            save_selected_results(output_path, all_result_dic, selected_means, selected_vars, selected_means_allreps,
                                  selected_vars_allreps, chosen_metrics, reward_types, ridx)

    if len(reps) > 1:
        save_selected_results_allreps(output_path, selected_means_allreps, selected_vars_allreps, chosen_metrics,
                                      reward_types, len(reps))











