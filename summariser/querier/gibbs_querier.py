import random
from summariser.querier.random_querier import RandomQuerier
from summariser.utils.misc import softmaxSample

class GibbsQuerier(RandomQuerier):

    def getQuery(self,log):
        mix_values = self.getMixReward()

        sum_idx1 = softmaxSample(mix_values,1.0)
        sum_idx2 = softmaxSample(mix_values,-1.0)

        ### ensure the sampled pair has not been queried before
        while(self.inLog(sum_idx1,sum_idx2,log)):
            sum_idx1 = softmaxSample(mix_values,1.0)
            sum_idx2 = softmaxSample(mix_values,-1.0)

        if random.random() > 0.5:
            return sum_idx1, sum_idx2
        else:
            return sum_idx2, sum_idx1
