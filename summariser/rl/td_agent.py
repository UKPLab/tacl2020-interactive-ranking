import sys
import os

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('./'))

#from summariser.algorithms._summarizer import Summarizer
from summariser.utils.corpus_reader import CorpusReader
from summariser.vector.state_type import State
from summariser.rouge.rouge import Rouge
#from summariser.utils.summary_samples_reader import *
from summariser.utils.misc import softmaxSample
from resources import PROCESSED_PATH
from summariser.vector.vector_generator import Vectoriser
from summariser.utils.reader import readSummaries
from summariser.utils.evaluator import evaluateSummary
from summariser.utils.misc import aggregateScores,addResult,bellCurvise

import numpy as np
import random


class TDAgent:
    def __init__(self, vectoriser, summaries, train_round=1000, strict_para=3.):

        # hyper parameters
        self.gamma = 1.
        self.epsilon = 1.
        self.alpha = 0.001
        self.lamb = 1.0

        # training options and hyper-parameters
        self.train_round = train_round
        self.strict_para = strict_para

        # samples
        self.summaries = summaries
        self.vectoriser = vectoriser
        self.softmax_list = []
        self.weights = np.zeros(self.vectoriser.vec_length)


    def __call__(self,reward_list):
        summary = self.trainModel(self.summaries, reward_list )
        return summary


    def trainModel(self, summary_list, reward_list):

        for ii in range(int(self.train_round)):

            if (ii+1)%1000 == 0 and ii!= 0:
                print('trained for {} episodes.'.format(ii+1))

            self.alpha = 0.001

            #find a new sample, using the softmax value
            idx = softmaxSample(reward_list,self.strict_para,self.softmax_list)

            # train the model for one episode
            self.train(summary_list[idx],reward_list[idx])

        summary = self.produceSummary()
        return ' '.join(summary)

    def produceSummary(self):
        state = State(self.vectoriser.sum_token_length, self.vectoriser.base_length, len(self.vectoriser.sentences),
                      self.vectoriser.block_num, self.vectoriser.language)

        # select sentences greedily
        while state.terminal_state == 0:
            new_sent_id = self.getGreedySent(state)
            if new_sent_id == 0:
                break
            else:
                state.updateState(new_sent_id - 1, self.vectoriser.sentences)

        # if the selection terminates by 'finish' action
        if new_sent_id == 0:
            assert len(''.join(state.draft_summary_list).split(' ')) <= self.vectoriser.sum_token_length
            return state.draft_summary_list

        # else, the selection terminates because of over-length; thus the last selected action is deleted
        else:
            return state.draft_summary_list[:-1]

    def coreUpdate(self, reward, current_vec, new_vec, vector_e):
        delta = reward + np.dot(self.weights, self.gamma * new_vec - current_vec)
        vec_e = self.gamma * self.lamb * vector_e + current_vec
        self.weights = self.weights + self.alpha * delta * vec_e
        return vec_e

    def getGreedySent(self, state):
        highest_value = float('-inf')
        best_sent_id = -1
        for act_id in state.available_sents:

            # for action 'finish', the reward is the terminal reward
            if act_id == 0:
                temp_state_value = np.dot(self.weights,
                                          state.getSelfVector(self.vectoriser.top_ngrams_list,
                                                                self.vectoriser.sentences))

            # otherwise, the reward is 0, and value-function can be computed using the weight
            else:
                temp_state_vec = state.getNewStateVec(act_id-1, self.vectoriser.top_ngrams_list,
                                                      self.vectoriser.sentences)
                temp_state_value = np.dot(self.weights, temp_state_vec)

            # get action that results in highest values
            if temp_state_value > highest_value:
                highest_value = temp_state_value
                best_sent_id = act_id

        # the return value ranges from 0 to act_num+1 (inclusive)
        return best_sent_id

    def train(self,summary_actions, summary_value):
        state = State(self.vectoriser.sum_token_length, self.vectoriser.base_length,
                      len(self.vectoriser.sentences), self.vectoriser.block_num, self.vectoriser.language)
        current_vec = state.getStateVector(state.draft_summary_list, state.historical_actions,
                                           self.vectoriser.top_ngrams_list, self.vectoriser.sentences)

        vector_e = np.zeros(self.vectoriser.vec_length)

        for act in summary_actions:
            #BE CAREFUL: here the argument for updateState is act, because here act is the real sentence id, not action
            reward = state.updateState(act, self.vectoriser.sentences, True)
            new_vec = state.getStateVector(state.draft_summary_list, state.historical_actions,
                                           self.vectoriser.top_ngrams_list,self.vectoriser.sentences)
            vector_e = self.coreUpdate(reward, current_vec, new_vec, vector_e)
            current_vec = new_vec
            del new_vec

        new_vec = np.zeros(self.vectoriser.vec_length)
        self.coreUpdate(summary_value, current_vec, new_vec, vector_e)


if __name__ == '__main__':

    dataset = 'DUC2001'
    #reward_type = 'rouge-personal' #'js12gram'
    reward_type = 'heuristic' #'js12gram'
    episode = 500
    rl_strict = 5.

    # read documents and ref. summaries
    reader = CorpusReader(PROCESSED_PATH)
    data = reader.get_data(dataset)

    topic_cnt = 0
    all_result_dic = {}

    for topic, docs, models in data:
        topic_cnt += 1
        print('\n{} {}th TOPIC: {}'.format(dataset,topic_cnt,topic))

        vec = Vectoriser(docs)
        summaries, rewards = readSummaries(dataset,topic,reward_type.split('-')[0])
        rl_agent = TDAgent(vec,summaries,episode,rl_strict)
        if 'personal' not in reward_type:
            if 'aggregate' in reward_type:
                rewards = aggregateScores(rewards)
            summary = rl_agent(rewards)
            print('summary length : {}'.format(len(summary.split(' '))))

        for model in models:
            model_name = model[0].split('/')[-1].strip()
            if 'personal' in reward_type:
                summary = rl_agent(rewards[model_name])
                print('summary length : {}'.format(len(summary.split(' '))))
            result = evaluateSummary(summary,model)
            print('---model {}---'.format(model_name))
            for metric in result:
                print('{} : {}'.format(metric,result[metric]))
            addResult(all_result_dic,result)

        print('\n=====UNTIL TOPIC {}, REWARD TYPE {}, EPISODE {}, STRICT {}====='.format(topic_cnt,reward_type,episode,rl_strict))
        for metric in all_result_dic:
            print('{} : {}'.format(metric,np.mean(all_result_dic[metric])))





