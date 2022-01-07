import numpy as np
from gppl.gp_pref_learning import pref_likelihood
from summariser.querier.pairwise_uncertainty_querier import PairUncQuerier


class InformationGainQuerier(PairUncQuerier):
    '''
    Choose the pair with highest expected information gain over the score function given the pairwise label.
    It computes the EIG using its symmetrical property to take the information gain over the pairwise label given the
    score function. This is not computable in closed form as the non-linearity makes the expectation over score
    function values intractable. Hence we use monte-carlo sampling.

    We could also calculate the uncertainty in the pairwise label probabilities. This also requires a similar monte-carlo
    sampling step. The downside of that is that it would not consider how much this uncertainty would actually be
    reduced by the chosen label. Furthermore, the stage2 of the system requires the learned reward function not the
    pairwise labels, so learning the preference function is more direct.

    Similar to uncertainty sampling, in that we choose uncertain pairs, except
    that here we only consider uncertainty in the score functions, not uncertainty in the
    preference labels where summaries have similar scores.

    The approach implemented below is similar to "pure exploration" from González, J., Dai, Z., Damianou, A.,
    & Lawrence, N. D. (2017, July). Preferential Bayesian Optimization. In International Conference on Machine Learning
    (pp. 1282-1291).
    The difference is that they chose the pair with the highest variance in p(preference label).
    Here we maximise expected information gain. I think this makes no difference unless the label
    noise is different for different pairs (e.g. if we consider asking different annotators), in which case IG should
    be used. However, IG is easier to compute.

    '''
    def _compute_pairwise_scores(self, f, Cov):
        #Get the current entropy of pairs
        var = np.diag(Cov)

        pairwise_var = var[:, None] + var[None, :] - 2 * Cov
        pair_probs = pref_likelihood(f, pairwise_var)

        neg_probs = 1 - pair_probs
        # stop warnings
        pair_probs[pair_probs < 1e-8] = 1e-8
        neg_probs[neg_probs < 1e-8] = 1e-8

        H_current = - pair_probs * np.log(pair_probs) - (neg_probs) * np.log(neg_probs)

        # now approximate the future entropy -- if there were no uncertainty in f -- a la Houlsby 2011 BALD method.
        # This looks like it may have some errors in compared to the Houlsby paper, but that only gives the
        # value for classification, not preference learning, and contains approximations... so it might work out?
        C = np.sqrt(2 * np.pi * np.log(2) / 2.0)
        H_updated = (-np.log(0.5)) * C / np.sqrt(C**2 + pairwise_var) * np.exp(- (f-f.T)**2 / (2*(C*2 + pairwise_var)))

        IG = H_current - H_updated
        return IG



