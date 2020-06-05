import sys
import os

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('./'))

from summariser.utils.corpus_reader import CorpusReader
from summariser.vector.state_type import State
#from summariser.utils.summary_samples_reader import *
from summariser.utils.misc import softmaxSample
from resources import PROCESSED_PATH
from summariser.vector.vector_generator import Vectoriser
from summariser.utils.evaluator import evaluateSummary
from summariser.utils.reader import readSummaries
from summariser.utils.misc import aggregateScores,addResult,bellCurvise

import numpy as np
import random

import torch
from torch.autograd import Variable


class DeepTDAgent:
    def __init__(self, vectoriser, summaries, train_round=5000, strict_para=3):

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
        self.weights = np.zeros(self.vectoriser.vec_length)

        # deep training
        self.hidden_layer_width = int(self.vectoriser.vec_length/2)


    def __call__(self,reward_list):
        self.softmax_list = []
        summary = self.trainModel(self.summaries, reward_list)
        return summary


    def trainModel(self, summary_list, reward_list):
        _,self.softmax_list = softmaxSample(reward_list,self.strict_para,[],True)

        self.deep_model = torch.nn.Sequential(
            torch.nn.Linear(self.vectoriser.vec_length, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, 1),
        )
        self.optimiser = torch.optim.Adam(self.deep_model.parameters())

        for ii in range(int(self.train_round)):

            if (ii+1)%1000 == 0 and ii!= 0:
                print('trained for {} episodes.'.format(ii+1))

            #find a new sample, using the softmax value
            idx = softmaxSample(reward_list,self.strict_para,self.softmax_list,False)

            # train the model for one episode
            loss = self.train(summary_list[idx],reward_list[idx])

        summary = self.produceSummary()
        return ' '.join(summary)

    def produceSummary(self):
        state = State(self.vectoriser.sum_token_length, self.vectoriser.base_length, len(self.vectoriser.sentences),self.vectoriser.block_num, self.vectoriser.language)

        # select sentences greedily
        while state.terminal_state == 0:
            new_sent_id = self.getGreedySent(state)
            if new_sent_id == 0:
                break
            else:
                state.updateState(new_sent_id-1, self.vectoriser.sentences)

        return state.draft_summary_list[:]

    def coreUpdate(self, reward, current_vec, new_vec, vector_e):
        delta = reward + np.dot(self.weights, self.gamma * new_vec - current_vec)
        vec_e = self.gamma * self.lamb * vector_e + current_vec
        self.weights = self.weights + self.alpha * delta * vec_e
        return vec_e

    def getGreedySent(self, state):
        if state.available_sents == [0]: return 0
        vec_list = []
        str_vec_list = []

        for act_id in state.available_sents:
            # for action 'finish', the reward is the terminal reward
            if act_id == 0:
                current_state_vec = state.getSelfVector(self.vectoriser.top_ngrams_list, self.vectoriser.sentences)
                vec_variable = Variable(torch.from_numpy(np.array(current_state_vec)).float())
                terminate_reward = self.deep_model(vec_variable.unsqueeze(0)).data.numpy()[0][0]

            # otherwise, the reward is 0, and value-function can be computed using the weight
            else:
                temp_state_vec = state.getNewStateVec(act_id-1, self.vectoriser.top_ngrams_list,
                                                      self.vectoriser.sentences)
                vec_list.append(temp_state_vec)
                str_vec = ''
                for ii,vv in enumerate(temp_state_vec):
                    if vv != 0.:
                        str_vec += '{}:{};'.format(ii,vv)
                str_vec_list.append(str_vec)

        # get action that results in highest values
        variable = Variable(torch.from_numpy(np.array(vec_list)).float())
        values = self.deep_model(variable)
        values_list = values.data.numpy()
        #print('vectors list: ')
        #for vv in str_vec_list:
            #print(vv)
        max_value = float('-inf')
        max_idx = -1
        for ii,value in enumerate(values_list):
            if value[0] > max_value:
                max_value = value[0]
                max_idx = ii

        if terminate_reward > max_value:
            return 0
        else:
            return state.available_sents[max_idx+1]


    def train(self,summary_actions, summary_value):
        state = State(self.vectoriser.sum_token_length, self.vectoriser.base_length,
                      len(self.vectoriser.sentences), self.vectoriser.block_num, self.vectoriser.language)
        current_vec = state.getStateVector(state.draft_summary_list, state.historical_actions,
                                           self.vectoriser.top_ngrams_list, self.vectoriser.sentences)

        vec_list = []
        vec_list.append(current_vec)

        for act in summary_actions:
            #BE CAREFUL: here the argument for updateState is act, because here act is the real sentence id, not action
            reward = state.updateState(act, self.vectoriser.sentences, True)
            new_vec = state.getStateVector(state.draft_summary_list, state.historical_actions,
                                           self.vectoriser.top_ngrams_list,self.vectoriser.sentences)
            vec_list.append(new_vec)

        return self.deepTrain(vec_list,summary_value)


    def deepTrain(self, vec_list, last_reward):
        value_variables = self.deep_model(Variable(torch.from_numpy(np.array(vec_list)).float()))
        value_list = value_variables.data.numpy()
        target_list = []
        for idx in range(len(value_list)-1):
            target_list.append(self.gamma*value_list[idx+1][0])
        target_list.append(last_reward)
        target_variables = Variable(torch.from_numpy(np.array(target_list)).float())

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(value_variables,target_variables)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()


if __name__ == '__main__':

    dataset = 'DUC2001'
    #reward_type = 'rouge-personal' #'js12gram'
    reward_type = 'heuristic' #'js12gram'
    episode = 2000
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
        rl_agent = DeepTDAgent(vec,summaries,episode,rl_strict)
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





