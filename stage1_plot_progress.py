'''

Plot the progress of the interactive learning process by showing metrics against no. interactions.

'''
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

matplotlib.rcParams.update({'font.size': 12})

tasks = ['duc2001']  # ['supert_duc2001', 'supert_bi_duc2001']  # 'bertcqa'  # 'coala' #  #
styles = ['-', ':', '-.', '--']
markers = ['x', 'v', '*', 's', 'd', 'p', 'o', '>']

method_str = {
    'random': 'rand.',
    'unc': 'UNC',
    'unpa': 'UNPA',
    'eig': 'EIG',
    'tp': 'TP',
    'imp': 'IMP'
}

method_tags = {
    'random': 'ran',
    'unc': 'unc',
    'unpa': 'unc',
    'eig': 'eig',
    'tp': 'tp',
    'imp': 'imp'
}

learners = ['gpplhh', 'lr']
learner_str = {
    'gpplhh': 'GPPL',
    'lr': 'BT'
}

learners.append('H')
learner_str['H'] = 'no interaction'

metric_str = {
    'pcc': "Pearson's r",
    'accuracy': 'Accuracy',
    'ndcg_at_1%': 'NDCG @1%',
    'ndcg_at_5%': 'NDCG @5%',
    'ndcg_at_10%': 'NDCG @10%'
}

ylimits = None
xticks = None

plt.figure()
# plt.figure(figsize=(8.0, 12.0))
# plt.subplot(212)

for t, task in enumerate(tasks):

    plt.gca().set_prop_cycle(None)

    if task == 'bertcqa' or task == 'supert_duc2001' or task == 'supert_bi_duc2001':
        methods = {
            'gpplhh': ['random', 'eig', 'imp'],
            'lr': ['random', 'unc'],
            'H': ['random']
        }
    else:
        methods = {
            'gpplhh': ['random', 'unpa', 'eig', 'tp', 'imp'],
            'lr': ['random', 'unc'],
            'H': ['random']
        }

    fallback_path = None  # where else to look if the results are not in the first output_path
    if task == 'coala':
        inters = [1, 3, 5, 7, 10, 15, 20, 25]  # 50, 100?
        xlimits = (0, 25)
        topics = ['cooking', 'travel', 'apple']
        metric = 'ndcg_at_5%'
        output_path = './results_coala/lno03_%s_%iinter_%s_rep%i/table_all_reps.csv'
        baseline_path = './results_coala/coala_H_%s_rep0/table_all_reps.csv'
    elif task == 'bertcqa':
        inters = [1, 5, 10, 15, 20]
        xlimits = (0, 22)
        xticks = [0, 5, 10, 15, 20]
        ylimits = (0.50, 0.78)
        topics = ['cooking', 'travel', 'apple']
        metric = 'ndcg_at_5%'
        output_path = './results_cqa/cqa_bert_%s_%s%s_%s_rep%i/table_all_reps.csv'
        baseline_path = './results_cqa/cqa_bert_H_%s_rep0/table_all_reps.csv'
    elif task == 'supert_duc2001':
        xlimits = (0, 100)
        inters = [10, 20, 50, 75, 100]  # need to copy results for 20, 50, and 75 from Apu to ./results
        metric = 'ndcg_at_1%'
        output_path = './results_1/duc01_supert_%s_%s%s_rep%i/table_all_reps.csv'
        fallback_path = './results/duc01_supert_%s_%s%s_rep%i/table_all_reps.csv'
    elif task == 'supert_bi_duc2001':
        xlimits = (0, 100)
        inters = [10, 20, 50, 75, 100]  # need to copy results for 20, 50, and 75 from Apu to ./results
        metric = 'ndcg_at_1%'
        output_path = './results/duc01_supert_bi_%s_%s%s_rep%i/table_all_reps.csv'
        baseline_path = './results/duc01_supert_H_rep0/table_all_reps.csv'
    else:
        inters = [10, 20, 50, 75, 100]  # need to copy results for 20, 50, and 75 from Apu to ./results
        xlimits = (0, 100)
        metric = 'ndcg_at_1%'
        output_path = './results_reaper_1/duc01_reaper_%s_%s%s_rep%i/table_all_reps.csv'
        baseline_path = './results_reaper_1/duc01_reaper_H_rep0/table_all_reps.csv'
        # output_path = './results/duc01_%s_%iinter2_rep%i/table_all_reps.csv'

    m = 0

    for learner in learners:
        for method in methods[learner]:
            my_results = []
            methodtag = method_tags[method]

            if learner == 'H':
                idx_last_rep = 0
            elif learner == 'gpplhh':
                if method == 'random' and task == 'bertcqa':
                    idx_last_rep = 4
                elif method == 'random' and (task == 'duc2001' or task == 'supert_duc2001' or task == 'supert_bi_duc2001'):
                    idx_last_rep = 9
                else:
                    idx_last_rep = 0
            else:
                if method == 'unc' and task == 'supert_duc2001':
                    idx_last_rep = 0
                else:
                    idx_last_rep = 9

            print('Learner: %s' % learner)

            for ninter in inters:
                outputdir = output_path

                if task == 'coala' or task == 'bertcqa':
                    # take an average over the topics
                    val = 0

                    for topic in topics:
                        if learner == 'H':
                            result_file = baseline_path % topic
                        elif ninter == 10:
                            result_file = outputdir % (methodtag, learner, '', topic, idx_last_rep)
                        else:
                            result_file = outputdir % (methodtag, learner, '_%i' % ninter, topic, idx_last_rep)

                        print('Reading data from %s' % result_file)
                        result_data = pd.read_csv(result_file, index_col=0, sep=',')
                        result_data = result_data[metric]
                        val += result_data[result_data.index == method]

                    val /= float(len(topics))

                elif task == 'duc2001' or task == 'supert_duc2001' or task == 'supert_bi_duc2001':
                    if learner == 'H':
                        if task == 'supert_duc2001':
                            continue

                        result_file = baseline_path
                    elif ninter == 100:
                        result_file = outputdir % (methodtag, learner, '', idx_last_rep)
                        if not os.path.exists(result_file) and task == 'supert_duc2001':
                            result_file = fallback_path % (methodtag, learner, '', idx_last_rep)
                            print('Falling back to path %s' % result_file )
                    else:
                        result_file = outputdir % (methodtag, learner, '_%i' % ninter, idx_last_rep)
                        if not os.path.exists(result_file) and task == 'supert_duc2001':
                            result_file = fallback_path % (methodtag, learner, '_%i' % ninter, idx_last_rep)
                            print('Falling back to path %s' % result_file )

                    result_data = pd.read_csv(result_file, index_col=0, sep=',')
                    result_data = result_data[metric]
                    if method == 'random' and not np.any(result_data.index == method):
                        result_file = outputdir % (learner, ninter, 9)
                        result_data = pd.read_csv(result_file, index_col=0, sep=',')
                        result_data = result_data[metric]

                    print(result_file)
                    print(method)
                    print(result_data.index)
                    val = result_data[result_data.index == method]

                my_results.append(val)

            method_label = method_str[method]
            if task == 'supert_duc2001':
                method_label = method_str[method] + ',SUPERT'
            elif task == 'supert_bi_duc2001':
                method_label = method_str[method] + ',bigram+'

            print(m)
            print(markers)
            print(styles)
            print(inters)
            print(my_results)

            if learner == 'H':
                if task == 'supert_duc2001':
                    continue
                plt.plot(inters, my_results, label='%s' % (learner_str[learner]),
                         ls='-', marker='.', color='black')
            else:
                plt.plot(inters, my_results, label='%s,%s' % (learner_str[learner], method_label),
                         ls=styles[(t if task == 'supert_duc2001' or task == 'supert_bi_duc2001' else m) % len(styles)],
                         marker=markers[m % len(markers)] )

            m += 1

        plt.legend(loc='best')  # plt.legend(loc='lower left', bbox_to_anchor=(-0.01, 0.9), ncol=2, borderaxespad=0)

        if not os.path.exists('./plots'):
            os.mkdir('./plots')

        plot_dir = './plots/progress_%s' % (task)
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        plt.ylabel(metric_str[metric])
        plt.xlabel('num. interactions')

        plt.xlim(xlimits)
        if task != 'coala':
            if ylimits is not None:
                plt.ylim(ylimits)
            if xticks is not None:
                plt.xticks(xticks)
        elif task == 'coala':
            plt.xlim(left=0)
            plt.ylim(bottom=plt.ylim()[0] - 0.02)

        plt.grid(True, axis='y')

    # save after the results of all tasks have been included
    plt.savefig(os.path.join(plot_dir, '%s.pdf' % metric))
