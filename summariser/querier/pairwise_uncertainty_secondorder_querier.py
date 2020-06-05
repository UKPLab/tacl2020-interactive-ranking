from summariser.querier.pairwise_uncertainty_querier import PairUncQuerier
import numpy as np
# from gp_pref_learning import pref_likelihood

class PairUncSOQuerier(PairUncQuerier):
    '''
    Designed to be used with methods with non-zero covariance.
    '''

    def _compute_pairwise_scores(self, f, Cov):
        var = np.diag(Cov)

        second_order_uncertainty = var[:, None] + var[None, :] - 2*Cov
        second_order_uncertainty[range(len(var)), range(len(var))] = -np.inf

        return second_order_uncertainty


