'''

Plot the progress of the interactive learning process by showing metrics against no. interactions.

'''
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

task = 'coala' # 'duc2001' #

styles = ['-', '-.', '--', ':']
markers = ['x', 'v', '*', 's', 'd', 'p', 'o', '>']

methods = {
    'gpplhh': ['random', 'pair_unc', 'eig', 'tig', 'imp'],
    'lr': ['random', 'unc']
}
method_str = {
    'random': 'random',
    'unc': 'UNC',
    'pair_unc': 'UNCPA',
    'eig': 'EIG',
    'tig': 'TP',
    'imp': 'IMP'
}

learners = ['gpplhh', 'lr']
learner_str = {
    'gpplhh': 'GPPL',
    'lr': 'BT'
}

metric_str = {
    'pcc': "Pearson's r",
    'accuracy': 'Accuracy',
    'ndcg_at_1%': 'NDCG @1%',
    'ndcg_at_5%': 'NDCG @5%',
    'ndcg_at_10%': 'NDCG @10%'
}

if task == 'coala':
    inters = [1, 3, 5, 7, 10, 15, 20, 25] # 50, 100?
    topics = ['cooking', 'travel', 'apple']
    metrics = ['accuracy', 'ndcg_at_5%', 'pcc']
    output_path = './results_coala/lno03_%s_%iinter_%s_rep%i/table_all_reps.csv'
else:
    inters = [10, 20, 50, 75, 100] # need to copy results for 20, 50, and 75 from Apu to ./results
    metrics = ['ndcg_at_1%', 'pcc']

    output_path = './results/duc01_%s_%iinter2_rep%i/table_all_reps.csv'

for metric in metrics:

    plt.figure()

    m = 0

    for learner in learners:

        if learner == 'gpplhh':
            idx_last_rep = 0
        else:
            idx_last_rep = 9

        for method in methods[learner]:
            my_results = []

            for ninter in inters:

                if task == 'coala':
                    # take an average over the topics
                    val = 0

                    for topic in topics:
                        result_file = output_path % (learner, ninter, topic, idx_last_rep)
                        print('Reading data from %s' % result_file)
                        result_data = pd.read_csv(result_file, index_col=0, sep=',')
                        result_data = result_data[metric]
                        val += result_data[result_data.index == method]

                    val /= float(len(topics))

                elif task == 'duc2001':
                    result_file = output_path % (learner, ninter, idx_last_rep)
                    result_data = pd.read_csv(result_file, index_col=0, sep=',')
                    result_data = result_data[metric]
                    if method == 'random' and not np.any(result_data.index == method):
                        result_file = output_path % (learner, ninter, 9)
                        result_data = pd.read_csv(result_file, index_col=0, sep=',')
                        result_data = result_data[metric]

                    val = result_data[result_data.index == method]

                my_results.append(val)

            plt.plot(inters, my_results, label='%s,%s' % (learner_str[learner], method_str[method]),
                     ls=styles[m % len(styles)], marker=markers[m % len(markers)] )

            m += 1

    plt.legend(loc='best')

    plot_dir = './plots/progress_%s' % (task)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plt.ylabel(metric_str[metric])
    plt.xlabel('num. interactions')

    if task == 'coala':
        plt.xlim(left=0)
        plt.ylim(bottom=plt.ylim()[0] - 0.02)

    plt.grid(True, axis='y')

    plt.savefig(os.path.join(plot_dir, '%s.pdf' % metric))
