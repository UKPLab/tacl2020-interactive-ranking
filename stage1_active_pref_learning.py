import datetime
import json
import os
import sys
from datetime import datetime
import pandas as pd
from obtain_supert_scores import SupertVectoriser
from summariser.oracle.lno_ref_values import SimulatedUser
from summariser.querier.expected_improvement_querier import ExpectedImprovementQuerier
from summariser.querier.expected_information_querier import InformationGainQuerier
from summariser.querier.gibbs_querier import GibbsQuerier
from summariser.querier.pairwise_uncertainty_querier import PairUncQuerier
from summariser.querier.pairwise_uncertainty_secondorder_querier import PairUncSOQuerier
from summariser.querier.thompson_querier import ThompsonTopTwoQuerier, ThompsonInformationGainQuerier
from summariser.querier.uncertainty_querier import UncQuerier
from summariser.utils.corpus_reader import CorpusReader
from resources import PROCESSED_PATH
from summariser.utils.reader import readSampleSummaries
from summariser.vector.vector_generator import Vectoriser
from summariser.utils.evaluator import evaluateReward
from summariser.querier.random_querier import RandomQuerier
import numpy as np
import logging
from random import seed
logging.basicConfig(level=logging.DEBUG)
from summariser.querier.logistic_reward_learner import LogisticRewardLearner
from summariser.querier.GPPL_reward_learner import GPPLRewardLearner, GPPLHRewardLearner, GPPLHsRewardLearner


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

    if len(args) > 12:
        res_dir = args[12]
    else:
        res_dir = 'results'

    if len(args) > 5 and args[5][0] != '-':
        root_dir = args[5]
        if not os.path.exists(os.path.join(root_dir, res_dir)):
            os.mkdir(os.path.join(root_dir, res_dir))
        if not os.path.exists(os.path.join(root_dir, 'data')):
            os.mkdir(os.path.join(root_dir, 'data'))
    else:
        root_dir = '.'

    if len(args) > 6:
        nthreads = int(args[6])
    else:
        nthreads = 0

    if len(args) > 7:
        dataset = args[7]
    else:
        dataset = None

    if len(args) > 8:
        n_inter_rounds = int(args[8])
    else:
        n_inter_rounds = n_debug if n_debug else 100

    if len(args) > 9:
        feature_type = args[9] # can be april or supert
    else:
        feature_type = 'april'

    if len(args) > 10:
        rate = float(args[10])
    else:
        rate = 200

    if len(args) > 11:
        lspower = float(args[11])
    else:
        lspower = 1

    if len(args) > 13:
        temp = float(args[13])
    else:
        temp = 2.5

    if learner_type_str == 'LR':
        if querier_types is None:
            querier_types = ['random', 'unc']
        post_weight = 0.5 # 0.7 if n_inter_rounds == 100 else 0.3  # trade off between the heuristic rewards and the pref-learnt rewards
        n_reps = 10

    elif learner_type_str == 'GPPL':
        if querier_types is None:
            querier_types = [
                'random',
                'pair_unc',
                'pair_unc_SO',
            ]  # ['ttt', 'tp', 'imp']# 'ttt' 'random' 'gibbs' 'unc' 'eig' 'tp' 'imp' 'eig'

        post_weight = 0.5 # 0.7 if n_inter_rounds == 100 else 0.3
        n_reps = 5

    elif learner_type_str == 'GPPLH':
        if querier_types is None:
            querier_types = [
                'random',
                'pair_unc',
                'pair_unc_SO',
            ]
        post_weight = 1
        n_reps = 10

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

        n_reps = 1

        if len(querier_types) == 1 and querier_types[0] == 'random':
            n_reps = 10
            print('Using %i repeats because of random sampling.' % n_reps)
        else:
            print('Changing the number of repeats to 1 as there are no random initialisations with this method.')

    elif learner_type_str == 'H':
        post_weight = 0
        querier_types = ['random']
        n_reps = 1

    first_rep = 0
    reps = np.arange(first_rep, n_reps)

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

    return learner_type, learner_type_str, n_inter_rounds, output_folder_name_in, querier_types, root_dir, res_dir, \
           post_weight, reps, seeds, n_debug, nthreads, dataset, feature_type, rate, lspower, temp


def learn_model(topic, model, ref_values_dic, querier_type, learner_type, learner_type_str, summary_vectors, heuristics_list,
                post_weight, n_inter_rounds, all_result_dic, n_debug, output_path, n_threads, temp=2.5):
    model_name = model[0].split('/')[-1].strip()
    print('\n---ref. summary {}---'.format(model_name))

    rouge_values = ref_values_dic[model_name]
    if n_debug:
        rouge_values = rouge_values[:n_debug]

    if learner_type is not None:
        learner_type_label = learner_type.__name__
    else:
        learner_type_label = 'nolearner'

    reward_file = output_path + '/rewards_%s_%s_%s_%s.json' % (topic, model_name,
                                                               querier_type, learner_type_label)
    # if this has already been done, skip it!
    if os.path.exists(reward_file):
        print('Reloading previously computed results.')
        # reload the pre-computed rewards
        with open(reward_file, 'r') as fh:
            learnt_rewards = json.load(fh)
    else:
        oracle = SimulatedUser(rouge_values, m=temp)

        if querier_type == 'gibbs':
            querier = GibbsQuerier(learner_type, summary_vectors, heuristics_list, post_weight, rate, lspower)
        elif querier_type == 'unc':
            querier = UncQuerier(learner_type, summary_vectors, heuristics_list, post_weight, rate, lspower)
        elif querier_type == 'pair_unc':
            querier = PairUncQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads,
                                     rate, lspower)
        elif querier_type == 'pair_unc_SO':
            querier = PairUncSOQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads, rate,
                                       lspower)
        elif querier_type == 'imp':
            querier = ExpectedImprovementQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads,
                                                 rate, lspower)
        elif querier_type == 'eig':
            querier = InformationGainQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads,
                                             rate, lspower)
        elif querier_type == 'ttt':
            querier = ThompsonTopTwoQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads,
                                            rate, lspower)
        elif querier_type == 'tig' or querier_type == 'tp':
            querier = ThompsonInformationGainQuerier(learner_type, summary_vectors, heuristics_list, post_weight,
                                                     n_threads, rate, lspower)
        else:
            querier = RandomQuerier(learner_type, summary_vectors, heuristics_list, post_weight, n_threads,
                                    rate, lspower)

        log = []

        if 'tune' in learner_type_str:
            querier.tune_learner()

        if learner_type_str == 'GPPLHH' or learner_type_str == 'GPPLHHtune' \
                or learner_type_str == 'GPPLHHs':
            # the first sample should not use the default of random selection, but should already apply
            # the chosen AL strategy
            querier.random_initial_sample = False

        if learner_type_str != 'H': # heuristics only, no learning
            for round in range(n_inter_rounds):
                sum1, sum2 = querier.getQuery(log)
                pref = oracle.getPref(sum1, sum2)
                log.append([[sum1, sum2], pref])
                if querier_type != 'random' or round == n_inter_rounds - 1:
                    # with random querier, don't train until the last iteration as the intermediate results are not used
                    querier.updateRanker(log)

        print('Active learning complete. Now getting mixed rewards')
        learnt_rewards = querier.getMixReward()

        print('Saving the rewards for this model...')
        with open(reward_file, 'w') as fh:
            json.dump(learnt_rewards, fh)

    print('Computing metrics...')
    metrics_dic = evaluateReward(learnt_rewards, rouge_values)

    # learnt_reward = querier.getMixReward()
    # rmse,temp,cee = plotAgreement(np.array(rouge_values),np.array(learnt_reward),plot=False)
    # metrics_dic['lno-rmse'] = rmse
    # metrics_dic['lno-temperature'] = temp
    # metrics_dic['lno-cee'] = cee

    for metric in metrics_dic:
        print('metric {} : {}'.format(metric, metrics_dic[metric]))
        if metric in all_result_dic:
            all_result_dic[metric].append(metrics_dic[metric])
        else:
            all_result_dic[metric] = [metrics_dic[metric]]

    # print('Saving the model...')
    #
    # with open(output_path + 'learner_%s_%s_%s_%s.pkl' % (topic, model_name, querier_type,
    #                                                      learner_type.__name__), 'wb') as fh:
    #     pickle.dump(querier.reward_learner, fh)

    return learnt_rewards


def load_summary_vectors(summaries, dataset, topic, root_dir, docs, feature_type):
    summary_vecs_cache_file = root_dir + '/data/summary_vectors/%s/summary_vectors_%s_%s.csv' % (feature_type,
                                                                                                 dataset, topic)

    if not os.path.exists(root_dir + '/data/summary_vectors/%s' % feature_type):
        os.mkdir(root_dir + '/data/summary_vectors/%s' % feature_type)
    if os.path.exists(summary_vecs_cache_file):
        print('Warning: reloading feature vectors for summaries from cache')
        # This should be fine, but if there is an error, we may need to check that the loading order has not changed.
        summary_vectors = np.genfromtxt(summary_vecs_cache_file)

    elif feature_type == 'april' or feature_type == 'supertbigram+':
        vec = Vectoriser(docs)
        summary_vectors = vec.getSummaryVectors(summaries)
        np.savetxt(summary_vecs_cache_file, summary_vectors)

    elif feature_type == 'supert':
        vec = SupertVectoriser(docs)
        summary_vectors, _ = vec.getSummaryVectors(summaries, use_coverage_feats=True)
        np.savetxt(summary_vecs_cache_file, summary_vectors)
        print('Cached summary vectors to %s' % summary_vecs_cache_file)

    return summary_vectors


def save_result_dic(all_result_dic, output_path, rep, topic_cnt, querier_type, learner_type_str, n_inter_rounds):
    # Compute and save metrics for this topic
    print('=== (rep={}) RESULTS UNTIL TOPIC {}, QUERIER {}, LEARNER {}, INTER ROUND {} ===\n'.format(
        rep, topic_cnt, querier_type.upper(), learner_type_str, n_inter_rounds
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


def make_output_dir(root_dir, res_dir, output_folder_name, rep):
    if output_folder_name == -1:
        output_folder_name = datetime.datetime.now().strftime('started-%Y-%m-%d-%H-%M-%S')
    else:
        output_folder_name = output_folder_name + '_rep%i' % rep

    output_path = root_dir + '/' + res_dir + '/%s' % output_folder_name
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    return output_path


if __name__ == '__main__':

    '''
    Command line arguments:
    python stage1_active_pref_learning.py reward_learner_type n_debug output_folder_name querier_types

    reward_learner_type -- can be LR, GPPL (mixes the posterior with the heuristics), GPPLH (uses heuristic as a prior), 
    GPPLHH (uses heuristic as a prior and a heuristic to select the initial sample). The best performer so far is GPPLHH.

    n_debug -- set to 0 if you are not debugging; set to a higher number to select a subsample of the data for faster 
    debugging of the main setup.

    output_folder_name -- this name will be used to store your results (metrics) and the rewards produced by the learner.
    This will be a subfolder of ./results/ .

    querier_types -- a list of querier types. If the reward learner is LR, you can pass any subset of [random, unc].
    If the reward learner is any of the GPPL variants, you can pass [random, pair_unc, pair_unc_SO, tp, imp]. The best
    performers are tig and imp.    
    
    '''

    learner_type, learner_type_str, n_inter_rounds, output_folder_name, querier_types, root_dir, res_dir, post_weight, \
    reps, seeds, n_debug, n_threads, dataset, feature_type, rate, lspower, temp = process_cmd_line_args(sys.argv)

    # parameters
    if dataset is None:
        dataset = 'DUC2001'  # 'DUC2001'  # DUC2001, DUC2002, 'DUC2004'#

    print('Running stage1 summary preference learning with %s, writing to %s/%s/%s' % (
        dataset, root_dir, res_dir, output_folder_name))

    max_topics = -1  # set to greater than zero to use a subset of topics for debugging
    folders = []

    nqueriers = len(querier_types)
    chosen_metrics = ['ndcg_at_1%', 'pcc', 'tau', 'ndcg_at_5%', 'ndcg_at_10%', 'rho']

    selected_means_allreps = np.zeros((nqueriers, len(chosen_metrics)))
    selected_vars_allreps = np.zeros((nqueriers, len(chosen_metrics)))

    for rep in reps:

        selected_means = np.zeros((nqueriers, len(chosen_metrics)))
        selected_vars = np.zeros((nqueriers, len(chosen_metrics)))

        output_path = make_output_dir(root_dir, res_dir, output_folder_name, rep)

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

            # read documents and ref. summaries
            reader = CorpusReader(PROCESSED_PATH)
            data = reader.get_data(dataset)

            # store all results
            all_result_dic = {}
            topic_cnt = 0

            for topic, docs, models in data:

                print('\n=====(repeat {}) TOPIC {}, QUERIER {}, INTER ROUND {}====='.format(rep, topic,
                                                                  querier_type.upper(), n_inter_rounds))

                topic_cnt += 1
                if 0 < max_topics < topic_cnt or (n_debug and topic_cnt > 1):
                    continue

                summaries, ref_values_dic, heuristic_list = readSampleSummaries(dataset, topic, feature_type)
                print('num of summaries read: {}'.format(len(summaries)))

                summary_vectors = load_summary_vectors(summaries, dataset, topic, root_dir, docs, feature_type)

                if n_debug:
                    heuristic_list = heuristic_list[:n_debug]
                    summary_vectors = summary_vectors[:n_debug]

                for model in models:
                    learnt_rewards = learn_model(
                        topic, model, ref_values_dic, querier_type, learner_type, learner_type_str, summary_vectors,
                        heuristic_list, post_weight, n_inter_rounds, all_result_dic, n_debug, output_path, n_threads,
                        temp
                    )
                    # best summary idx
                    bestidx = np.argmax(learnt_rewards)
                    sentidxs = summaries[bestidx]
                    sentcount = 0
                    summary_sents = {}
                    for doc_id, doc in enumerate(docs):
                        _, doc_sents = doc
                        for sent_text in doc_sents:
                            if sentcount in sentidxs:
                                summary_sents[sentcount] = sent_text

                            sentcount += 1
                    summary_text = ""
                    for sent in sentidxs:
                        summary_text += summary_sents[sent]

                    print('SUMMARY: ')
                    print(summary_text)

                if n_debug:
                    heuristic_list = heuristic_list[:n_debug]
                    summary_vectors = summary_vectors[:n_debug]

                for model in models:
                    learn_model(topic, model, ref_values_dic, querier_type, learner_type, learner_type_str,
                                summary_vectors, heuristic_list, post_weight, n_inter_rounds, all_result_dic, n_debug,
                                output_path, n_threads, temp=temp)

                save_result_dic(all_result_dic, output_path, rep, topic_cnt, querier_type, learner_type_str,
                                n_inter_rounds)

            save_selected_results(output_path, all_result_dic, selected_means, selected_vars, selected_means_allreps,
                                  selected_vars_allreps, chosen_metrics, querier_types, qidx)

    save_selected_results_allreps(output_path, selected_means_allreps, selected_vars_allreps, chosen_metrics,
                                      querier_types, len(reps))
