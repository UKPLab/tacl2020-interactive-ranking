from gppl.gp_pref_learning import pref_likelihood
from scipy.stats import norm
import numpy as np
from summariser.querier.pairwise_uncertainty_querier import PairUncQuerier


class ThompsonTopTwoQuerier(PairUncQuerier):
    '''
    Uses Thompson sampling to sample the scores for the summaries, then selects the two summaries with highest total
    score. Pairs that were already compared are ignored.

    The idea is that this is an exploitative approach (learns about the current best estimated item) but by doing
    Thompson sampling, there is also some exploration.

    Downside: it doesn't check whether the selected pair is actually informative. Scenarios with lots of very similar,
    very good items may over-exploit.

    '''

    def _get_candidates(self):

        f = self.reward_learner.get_rewards()
        var = self.reward_learner.predictive_var()

        # sample the scores from the distributions
        self.f_sample = norm.rvs(f, np.sqrt(var))  #mvn.rvs(f, Cov) --> not necessary to consider covariance here

        # consider only the top most uncertain items
        num = 20
        candidate_idxs = np.argsort(self.f_sample)[-num:]

        self.f_sample = self.f_sample[candidate_idxs]

        return candidate_idxs

    def _compute_pairwise_scores(self, f, Cov):

        # find the total scores for the pairs, excluding any pairs in the logs
        f_total = self.f_sample[:, None] + self.f_sample[None, :]

        return f_total


class ThompsonInformationGainQuerier(ThompsonTopTwoQuerier):
    '''
    Selects (1) the best item, x_1, using thompson sampling,
    then (2) selects the best pair to learn about this item using information gain over f(x_1).

    Motivation: learn about whether the current best item is really the best.

    Doing this as two steps avoids selecting useless items with high information gain.
    By considering information gain, we avoid uninformative comparisons between good items (which may occur in
    Thompson Top Two or Expected Improvement) -- it will favour comparing the best item with good items with low
    uncertainty; contrast with TTT, which will favour comparing the best item with the next best item, but ignores
    uncertainty. By using Thompson sampling instead of maximum expected improvement,
    we also avoid the pitfall of only considering the current next-best item at each
    iteration. TS does some exploration, effectively considering IG/uncertainty sampling as part of the first step.

    The downside: we ignore the expected improvement of the other chosen item.
    It's not simple to compute improvement for the pair as it's not a simple sum (e.g. we replace probability of
    improvement for one item with 1 - probability of no improvement in either item). If we do this, then how
    do we account for information gain? It cannot be a separate step as we already select over pairs, but multiplying
    by improvement in step (1) would give a greedy bias toward items with high IG despite low improvement.

    We could do this: step (1) as before, so that we learn about best alternatives to current best. In step (2),
    choose based on normalised_IG(x_1) * improvement(x_1)  + normalised_IG(x_2) * improvement(x_2).
    The original suggestion would ignore the x_2 terms.
    Danger: not clear whether this formula is truly principled or if we need to learn some term weightings to make it work.
    Could end up choosing pairs that help to learn x_2, not x_1, which means we learn very slowly about the items
    with the best chance of improvement.

    The approach implemented below is similar to "dueling Thompson sampling" from González, J., Dai, Z., Damianou, A.,
    & Lawrence, N. D. (2017, July). Preferential Bayesian Optimization. In International Conference on Machine Learning
    (pp. 1282-1291).
    The difference is firstly in the first step: we sample the underlying preference function, not the pairwise
    difference function. This means we don't then need to integrate across possible pairs to find the top candidate
    summary.
    Secondly, in how the second step is calculated: they chose the pair with the highest variance in
    p(preference label). Here we maximise expected information gain. I think this makes no difference unless the label
    noise is different for different pairs (e.g. if we consider asking different annotators), in which case IG should
    be used. However, IG is easier to compute.


    '''

    def _get_candidates(self):

        f = self.reward_learner.get_rewards()
        var = self.reward_learner.predictive_var()

        # sample the scores from the distributions
        self.f_sample = norm.rvs(f, np.sqrt(var))  #mvn.rvs(f, Cov) --> not necessary to consider covariance here

        # consider only the top most uncertain items
        num = 20**2
        candidate_idxs = np.argsort(self.f_sample)[-num:]

        self.f_sample = self.f_sample[candidate_idxs]

        return candidate_idxs

    def _compute_pairwise_scores(self, f, Cov):
        best_idx = np.argmax(self.f_sample)  # chooses the best item in the sample

        var = np.diag(Cov)

        v_idxs = np.zeros(len(f), dtype=int) + best_idx
        u_idxs = np.arange(len(f), dtype=int)

        var_cands = var + var[best_idx] - 2 * Cov[:, best_idx]
        pair_probs = pref_likelihood(f, var_cands, v=v_idxs, u=u_idxs)

        neg_probs = 1 - pair_probs
        # stop warnings
        pair_probs[pair_probs < 1e-8] = 1e-8
        neg_probs[neg_probs < 1e-8] = 1e-8

        H_current = - pair_probs * np.log(pair_probs) - (neg_probs) * np.log(neg_probs)

        # now approximate the future entropy -- if there were no uncertainty in f -- a la Houlsby 2011 BALD method.
        C = np.sqrt(2 * np.pi * np.log(2) / 2.0)
        z_mean = f-f[best_idx]
        H_updated = (-np.log(0.5)) * C / np.sqrt(C**2 + var_cands) * np.exp(- z_mean**2 / (2*(C**2 + var_cands)))

        IG = H_current - H_updated[:, None]

        # make it back into a matrix
        IG_mat = np.zeros((f.size, f.size))
        IG_mat[best_idx, :] = IG.flatten()

        return IG_mat



