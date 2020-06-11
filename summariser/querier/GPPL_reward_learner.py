'''
Wrapper for Gaussian process preference learning (GPPL) to learn the latent reward function from pairwise preference
labels expressed by a noisy labeler.
'''
import logging

import numpy as np

import sys

from gppl.gp_classifier_vb import compute_median_lengthscales
from gppl.gp_pref_learning import GPPrefLearning


class GPPLRewardLearner():

    def __init__(self, steep=1.0, full_cov=False, heuristics=None, n_threads=0):
        self.learner = None
        self.steep = steep

        self.n_labels_seen = 0

        self.n_iterations = -1

        self.full_cov = full_cov # whether to compute the full posterior covariance

        self.tune = False
        self.fixed_s = True

        self.mu0 = None

        self.n_threads = n_threads

    def train(self, pref_history, vector_list, true_scores=None, tr_items=None):
        '''

        :param pref_history: a list of objects of the form [[item_id0, item_id1], preference_label ], where
        preference_label is 0 to indicate that item 0 is preferred and 1 to indicate item 1.
        :param vector_list: list of feature vectors for the items
        :return: nowt.
        '''
        # self.n_iterations += 1
        # # only update the model once every 5 iterations
        # if np.mod(self.n_iterations, 5) != 0:
        #     return

        if self.tune:
            rates = [800, 1600, 3200, 6400, 12800]
        else:
            rates = [200]

        best_train_score = -np.inf
        best_rate = 200

        for rate in rates:

            if self.learner is None or self.tune: # needs the vectors to init
                new_items_feat = np.array(vector_list)

                logging.debug('Estimating lengthscales for %i features from %i items' %
                      (new_items_feat.shape[1], new_items_feat.shape[0]))

                ls_initial = compute_median_lengthscales(new_items_feat, multiply_heuristic_power=1.0)
                # Tested with random selection, a value of multiply_heuristic_power=1 is better than 0.5 by far.

                logging.debug('Estimated length scales.')

                self.learner = GPPrefLearning(len(vector_list[0]), shape_s0=1.0, rate_s0=rate, use_svi=True,
                                              ninducing=500, max_update_size=1000, kernel_combination='*',
                                              forgetting_rate=0.7, delay=1, fixed_s=self.fixed_s, verbose=True,
                                              ls_initial=ls_initial)

                # self.learner.set_max_threads(self.n_threads)

                logging.debug('Initialised GPPL.')

                # put these settings in to reduce the number of iterations required before declaring convergence
                self.learner.min_iter_VB = 1
                self.learner.conv_check_freq = 1
                self.learner.n_converged = 1
            else:
                new_items_feat = None # only pass in the feature vectors in the first iteration

            # use the heuristic mean only for debugging
            new_item_ids0 = [data_point[0][0] for data_point in pref_history]
            new_item_ids1 = [data_point[0][1] for data_point in pref_history]
            new_labels = np.array([1 - data_point[1] for data_point in pref_history]) # for GPPL, item 2 is preferred if label == 0

            # new_item_ids0 = []
            # new_item_ids1 = []
            # new_labels = []

            logging.debug('GPPL fitting with %i pairwise labels' % len(new_labels))

            self.learner.fit(new_item_ids0, new_item_ids1, new_items_feat, new_labels, optimize=False,
                             input_type='binary', use_median_ls=False, mu0=self.mu0[:, None])

            if self.tune:
                # Can't really use Pearson in a realistic setting because we don't have the oracle
                # train_score = pearsonr(self.learner.f[tr_items].flatten(), true_scores)[0]
                # print('Training Pearson r = %f' % train_score)

                train_score = self.learner.lowerbound()
                print('ELBO = %.5f' % train_score)

                if train_score > best_train_score:
                    best_train_score = train_score
                    best_model = self.learner
                    best_rate = rate
                    print('New best train score %f with rate_s0=%f' % (train_score, rate))

        print('GPPL fitting complete in %i iterations.' % self.learner.vb_iter)

        if self.tune:
            self.learner = best_model
            print('Best tuned model has rate_s=%f' % best_rate)
            with open('./results/tuning_results.csv', 'a') as fh:
                fh.writelines(['%i' % best_rate])
        if not self.fixed_s:
            print('Learned model has s=%f' % self.learner.s)
            with open('./results/learning_s_results.csv', 'a') as fh:
                fh.writelines(['%f' % self.learner.s])

        self.n_labels_seen = len(pref_history)

        self.rewards, self.reward_var = self.learner.predict_f(full_cov=False, reuse_output_kernel=True,
                                                               mu0_output=self.mu0[:, None])


    def get_rewards(self):
        return self.rewards.flatten()

    def predictive_var(self):
        return self.reward_var.flatten()

    def predictive_cov(self, idxs):
        if self.full_cov:
            return self.learner.predict_f(out_idxs=idxs, full_cov=True, reuse_output_kernel=False,
                   mu0_output=self.mu0[idxs] if not self.mu0 is None and not np.isscalar(self.mu0) else self.mu0)[1]
        else:
            if self.reward_var.shape[1] == 1:
                return self.reward_var[idxs]
            else:
                return np.diag(self.reward_var[idxs])

class GPPLHRewardLearner(GPPLRewardLearner):
    def __init__(self, steep=1.0, full_cov=False, heuristics=None, n_threads=0, heuristic_offset=0.0, heuristic_scale=1.0):

        super(GPPLHRewardLearner, self).__init__(steep, full_cov, n_threads=n_threads)

        minh = np.min(heuristics)
        maxh = np.max(heuristics)

        self.mu0 = (heuristics - minh) /(maxh - minh) - 0.5 #  * 2 * np.sqrt(200)
        self.mu0 = self.mu0 * heuristic_scale + heuristic_offset

class GPPLHsRewardLearner(GPPLHRewardLearner):
    def __init__(self, steep=1.0, full_cov=False, heuristics=None, n_threads=0):

        super(GPPLHsRewardLearner, self).__init__(steep, full_cov, heuristics, n_threads)

        self.fixed_s = False
