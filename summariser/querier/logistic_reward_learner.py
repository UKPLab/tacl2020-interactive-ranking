import numpy as np
from sklearn import linear_model

from summariser.querier.GPPL_reward_learner import reduce_dimensionality


class LogisticRewardLearner():

    def __init__(self, steep=1.0, heuristics=None, n_threads=0, rate=200, lspower=1, do_dim_reduction=False):
        self.learner = linear_model.LogisticRegression(fit_intercept=False)
        self.steep = steep
        self.n_labels_seen = 0

        self.do_dim_reduction = do_dim_reduction

    def train(self,pref_history,vector_list):
        new_items_feat = np.array(vector_list)
        if self.do_dim_reduction and new_items_feat.shape[1] < 300:
            self.do_dim_reduction = False

        if self.do_dim_reduction:
            vector_list = reduce_dimensionality(new_items_feat)


        X = []
        Y = []
        for pref_idx in range(len(pref_history)):
            vec1 = np.array(vector_list[pref_history[pref_idx][0][0]])
            vec2 = np.array(vector_list[pref_history[pref_idx][0][1]])
            delta = self.steep*(vec1-vec2)
            preff = pref_history[pref_idx][1]
            X.append(delta)
            X.append(-delta)
            #the first summary is preferred
            if preff == 0:
                Y.append(1)
                Y.append(0)
            #the second summary is preferred
            elif preff == 1:
                Y.append(0)
                Y.append(1)

        X = np.array(X)
        Y = np.array(Y)

        self.learner.fit(X,Y)
        self.weights = np.array(self.learner.coef_[0])

        self.rewards = [np.dot(self.weights, vv) for vv in vector_list]

        self.n_labels_seen = len(pref_history)

    def get_rewards(self):
        return self.rewards

    def predictive_cov(self):
        # does not estimate covariance in its predictions
        return np.zeros(len(self.rewards))








