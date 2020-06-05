import random, numpy as np
from summariser.querier.random_querier import RandomQuerier
from summariser.utils.misc import sigmoid

class UncQuerier(RandomQuerier):
    '''
    Designed to be used with LR since the function for estimating the uncertainty scores was designed that way.
    '''

    def _getUncScores(self, scores):
        unc_scores = []

        for vv in scores:
            prob = sigmoid((vv-5)*.6)
            if prob > 0.5:
                unc_scores.append(2*(1-prob))
            else:
                unc_scores.append(2*prob)

        return np.array(unc_scores)

    def _getMostUncertainPair(self, unc_scores, log):

        # consider only the top N
        num = 20
        unc_idxs = np.argsort(unc_scores)[-num:]

        pair_unc = np.zeros((num, num))
        pair_unc += unc_scores[unc_idxs][:, None]
        pair_unc += unc_scores[unc_idxs][None, :]
        pair_unc[range(num), range(num)] = -np.inf

        for data_point in log:
            if data_point[0][0] not in unc_idxs or data_point[0][1] not in unc_idxs:
                continue
            dp0 = np.argwhere(unc_idxs == data_point[0][0]).flatten()[0]
            dp1 = np.argwhere(unc_idxs == data_point[0][1]).flatten()[0]
            pair_unc[dp0, dp1] = 0
            pair_unc[dp1, dp0] = 0

        selected = np.unravel_index(np.argmax(pair_unc), pair_unc.shape)
        selected = (unc_idxs[selected[0]], unc_idxs[selected[1]])

        return selected


    def getQuery(self,log):
        mix_values = self.getMixReward()
        unc_scores = self._getUncScores(mix_values)

        pair = self._getMostUncertainPair(unc_scores, log)

        if random.random() > 0.5:
            return pair[0], pair[1]
        else:
            return pair[1], pair[0]



