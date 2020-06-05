import json
import os
import sys
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from stage1_active_pref_learning import process_cmd_line_args, save_selected_results, save_selected_results_allreps

# import matplotlib
# from matplotlib.ticker import MultipleLocator
# matplotlib.use("Agg")
#
# import matplotlib.pyplot as plt
#
# def plot_metrics_per_topic(chosen_metrics, all_results, figs, avg_figs, nqueriers, qidx, querier_type, n_inter_rounds, learner_type_str):
#     for i, chosen_metric in enumerate(chosen_metrics):  # these are the metrics we really care about
#
#         results = np.mean(all_results[chosen_metric], axis=0)
#         ntopics = len(results) if not np.isscalar(results) else 1
#
#         colors = plt.rcParams['axes.prop_cycle'].by_key()[
#             'color']  # ['blue', 'red', 'green', 'yellow', 'orange', 'purple']
#
#         plt.figure(figs[i].number)
#         plt.bar(np.arange(ntopics) + (qidx / float(nqueriers + 1)), results, color=colors[qidx],
#                 width=1.0 / (nqueriers + 1.0), label=querier_type)
#
#         plt.figure(avg_figs[i].number)
#         plt.bar(qidx, np.mean(results), width=0.8, color=colors[qidx], label=querier_type, zorder=3)
#         plt.title('Performance after %i interactions with %s' % (n_inter_rounds, learner_type_str))
#
# def save_plots(figs, avg_figs, timestamp, chosen_metrics):
#     if not os.path.exists('./plots'):
#         os.mkdir('./plots')
#
#     plot_path = './plots/%s' % timestamp
#     if not os.path.exists(plot_path):
#         os.mkdir(plot_path)
#     for m, f in enumerate(figs):
#         plt.figure(f.number)
#         plt.legend(loc='lower left')
#         plt.savefig(os.path.join(plot_path, chosen_metrics[m] + '.pdf'))
#
#     for m, f in enumerate(avg_figs):
#         plt.figure(f.number)
#         plt.legend(loc='best')
#         plt.gca().yaxis.set_minor_locator(MultipleLocator(0.01))
#         plt.grid(True, 'minor', zorder=0)
#         plt.savefig(os.path.join(plot_path, chosen_metrics[m] + '_avg.pdf'))

if __name__ == '__main__':

    learner_type, learner_type_str, n_inter_rounds, output_folder_name, querier_types, root_dir, post_weight, reps, \
    seeds, n_debug, n_threads, dataset = process_cmd_line_args(sys.argv)

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

        output_folder_r = output_folder_name + '_rep%i' % rep
        output_path = root_dir + '/results/%s' % output_folder_r
        print('Checking %s' % output_path)

    # get a list of folders relating to this experiment
    folders = []
    with open('%s/folders.txt' % output_path, 'r') as fh:
        for folder_name in fh:
            folders.append(folder_name)

    nqueriers = len(querier_types)

    chosen_metrics = ['ndcg_at_1%', 'pcc', 'tau', 'ndcg_at_5%', 'ndcg_at_10%', 'rho', 'score_of_estimated_best']

    nreps = len(folders) # number of repeats that were actually completed

    selected_means_allreps = np.zeros((nqueriers, len(chosen_metrics)))
    selected_vars_allreps = np.zeros((nqueriers, len(chosen_metrics)))

    for r in range(nreps):
        output_path = folders[r].strip('\n')

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

            save_selected_results(output_path, all_result_dic, selected_means, selected_vars, selected_means_allreps,
                              selected_vars_allreps, chosen_metrics, querier_types, qidx)

    print('Saving result summary to %s' % output_path)
    save_selected_results_allreps(output_path, selected_means_allreps, selected_vars_allreps, chosen_metrics,
                                      querier_types, nreps)
