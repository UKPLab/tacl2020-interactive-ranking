import numpy as np
from summariser.querier.pairwise_uncertainty_querier import PairUncQuerier
from scipy.stats import norm

class ExpectedImprovementQuerier(PairUncQuerier):
    '''
    Compare the current best summary against the summary with the largest expected improvement
    over current best expected summary score.

    If the pair has already been seen, choose the next summary.

    If the best summary has already been compared against all others, do a random exploration
    step and replace the best summary in the pair with a random summary.

    The approach is basically the expected improvement method tested here: Sequential Preference-Based Optimization,
    Ian Dewancker, Jakob Bauer, Michael McCourt, Bayesian Deep Learning workshop at NIPS 2017.

    The approach implemented below is the related to "dueling Thompson sampling" and "copeland expected improvement"
    from González, J., Dai, Z., Damianou, A.,
    & Lawrence, N. D. (2017, July). Preferential Bayesian Optimization. In International Conference on Machine Learning
    (pp. 1282-1291).
    The difference to DTS: it uses expected improvement over the underlying preference function instead of thompson
    sampling from a pairwise difference function.
    Difference to CEI: we consider only the expected improvement over the underlying preference function scores
    as a first selection step, then compare to the best, rather than choosing pairs.

    '''

    def _get_candidates(self):
        f = self.reward_learner.get_rewards()

        # consider only the top ranked items.
        num = 20**2
        candidate_idxs = np.argsort(f)[-num:]

        return candidate_idxs

    def _compute_pairwise_scores(self, f, Cov):

        best_idx = np.argmax(f)
        f_best = f[best_idx]

        sigma = Cov[best_idx, best_idx] + np.diag(Cov) - Cov[best_idx, :] - Cov[:, best_idx]
        sigma[best_idx] = 1 # avoid the invalid value errors -- the value of u should be 0

        # for all candidates, compute u = (mu - f_best) / sigma
        u = (f - f_best) / np.sqrt(sigma) # mean improvement. Similar to preference likelihood, but that adds in 2
        # due to labelling noise
        cdf_u = norm.cdf(u) # probability of improvement
        pdf_u = norm.pdf(u) #

        E_improvement = sigma * (u * cdf_u + pdf_u)
        E_improvement[best_idx] = -np.inf

        # make it back into a matrix
        E_imp_mat = np.zeros((f.size, f.size))
        E_imp_mat[best_idx, :] = E_improvement

        return E_imp_mat




