import logging

from summariser.querier.random_querier import RandomQuerier
import numpy as np

import sys
from gppl.gp_pref_learning import pref_likelihood


class PairUncQuerier(RandomQuerier):
    '''
    Designed to be used with either GPPL or LR.
    '''
    def __init__(self, reward_learner_class, summary_vectors, heuristic_values, learnt_weight=0.5, n_threads=0,
                 rate=200, lspower=1, prior_scale=1.0, prior_offset=0.0, ):
        self.summary_vectors = summary_vectors

        if prior_scale != 1.0 or prior_offset != 0.0:
            self.reward_learner = reward_learner_class(full_cov=True, heuristics=heuristic_values, n_threads=n_threads,
                 heuristic_offset=prior_offset, heuristic_scale=prior_scale, rate=rate, lspower=lspower)
        else:
            self.reward_learner = reward_learner_class(full_cov=True, heuristics=heuristic_values, n_threads=n_threads,
                                                       rate=rate, lspower=lspower)

        self.heuristics = heuristic_values
        self.learnt_weight = learnt_weight
        self.learnt_values = [0.] * len(summary_vectors)
        
        self.random_initial_sample = True

    def _compute_pairwise_scores(self, f, Cov):
        var = np.diag(Cov)

        pair_probs = pref_likelihood(f, var[:, None] + var[None, :] - 2 * Cov)
        # stop warnings
        pair_probs[pair_probs < 1e-8] = 1e-8
        pair_probs[pair_probs > 1 - 1e-8] = 1 - 1e-8

        pairwise_entropy = - pair_probs * np.log(pair_probs) - (1 - pair_probs) * np.log(1 - pair_probs)
        pairwise_entropy[range(pair_probs.shape[0]), range(pair_probs.shape[1])] = -np.inf

        return pairwise_entropy

    def _get_candidates(self):
        v = self.reward_learner.predictive_var()

        # consider only the top most uncertain items
        num = 20
        candidate_idxs = np.argsort(v)[-num:]

        return candidate_idxs

    def getQuery(self, log):
        if self.reward_learner.n_labels_seen == 0 and self.random_initial_sample:
            return super().getQuery(log)
        elif self.reward_learner.n_labels_seen == 0:
            return self._get_good_and_dissimilar_pair()

        # get the current best estimate
        f = self.reward_learner.get_rewards()

        candidate_idxs = self._get_candidates()

        Cov = self.reward_learner.predictive_cov(candidate_idxs)

        pairwise_entropy = self._compute_pairwise_scores(f[candidate_idxs], Cov)

        # Find out which of our candidates have been compared already
        for data_point in log:
            if data_point[0][0] not in candidate_idxs or data_point[0][1] not in candidate_idxs:
                continue
            dp0 = np.argwhere(candidate_idxs == data_point[0][0]).flatten()[0]
            dp1 = np.argwhere(candidate_idxs == data_point[0][1]).flatten()[0]
            pairwise_entropy[dp0, dp1] = -np.inf
            pairwise_entropy[dp1, dp0] = -np.inf

        selected = np.unravel_index(np.argmax(pairwise_entropy), pairwise_entropy.shape)
        pe_selected = pairwise_entropy[selected[0], selected[1]]
        selected = (candidate_idxs[selected[0]], candidate_idxs[selected[1]])

        print('Chosen candidate: %i, vs. best: %i, with score = %f' %
              (selected[0], selected[1], pe_selected))

        return selected[0], selected[1]



