import numpy as np
import random
from summariser.utils.misc import normaliseList
from sklearn.metrics.pairwise import cosine_similarity


class RandomQuerier:

    def __init__(self, reward_learner_class, summary_vectors, heuristic_values, learnt_weight=0.5, n_threads=0,
                 rate=200, lspower=1):
        self.summary_vectors = summary_vectors
        if reward_learner_class is not None:
            self.reward_learner = reward_learner_class(heuristics=heuristic_values, n_threads=n_threads, rate=rate,
                                                       lspower=lspower)
        self.heuristics = heuristic_values
        self.learnt_weight = learnt_weight
        self.learnt_values = [0.]*len(summary_vectors)

        # use a random sample to initialise the AL process, then apply the AL strategy to subsequent iterations.
        # This flag only affects the classes that inherit from this class since the random querier always chooses
        # randomly
        self.random_initial_sample = True

    def tune_learner(self):
        print('Setting the reward learner to be tuned...')
        self.reward_learner.tune = True

    def inLog(self,sum1,sum2,log):
        for entry in log:
            if [sum1,sum2] in entry:
                return True
            elif [sum2,sum1] in entry:
                return True

        return False

    def _get_good_and_dissimilar_pair(self):
        # find two distinctive items as an initial sample. To limit complexity, first choose the item with
        # strongest heuristic:
        first_item = np.argmax(self.heuristics)

        # now compare this item to the others according to the feature vectors:
        sims = cosine_similarity(self.summary_vectors[first_item][None, :], self.summary_vectors)
        second_item = np.argmin(sims)

        return first_item, second_item

    def getQuery(self,log):
        if self.reward_learner.n_labels_seen == 0 and not self.random_initial_sample:
            return self._get_good_and_dissimilar_pair()

        summary_num = len(self.summary_vectors)
        rand1 = random.randint(0,summary_num-1)
        rand2 = random.randint(0,summary_num-1)

        ### ensure the sampled pair has not been queried before
        while rand2 == rand1 or self.inLog(rand1,rand2,log):
            rand1 = random.randint(0, summary_num-1)
            rand2 = random.randint(0, summary_num-1)

        return rand1, rand2

    def updateRanker(self,pref_log):
        if self.learnt_weight > 0:
            self.reward_learner.train(pref_log, self.summary_vectors)
            self.learnt_values = self.getReward()
        else:
            self.learnt_values = 0

    def getReward(self):
        values = self.reward_learner.get_rewards()
        return normaliseList(values)

    def getMixReward(self,learnt_weight=-1):
        if learnt_weight == -1:
            learnt_weight = self.learnt_weight

        mix_values = np.array(self.learnt_values)*learnt_weight + np.array(self.heuristics)*(1-learnt_weight)
        return normaliseList(mix_values)




