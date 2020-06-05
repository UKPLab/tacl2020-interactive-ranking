from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from summariser.utils.rank_metrics import ndcg_at_k
from summariser.utils.misc import *
from summariser.rouge.rouge import Rouge
from summariser.utils.misc import jsd
from resources import *
from collections import OrderedDict
import matplotlib.pyplot as plt

def evaluateReward(learnt_values, ref_values, short=False, top_answer=None):
    metrics_dic = OrderedDict()

    if not short:
        ### compute the absolute errors
        mse = mean_squared_error(ref_values,learnt_values)
        metrics_dic['mse'] = mse

        ### compute KL divergence
        #js = jsd(learnt_values,ref_values)
        #metrics_dic['jsd-original'] = js
        prob_optimal = getSoftmaxList(ref_values, 1.0)
        prob_learnt = getSoftmaxList(learnt_values, 1.0)
        js = jsd(prob_optimal,prob_learnt)
        metrics_dic['jsd-softmax'] = js
        #kld = stats.entropy(prob_optimal, prob_learnt)
        #metrics_dic['kld'] = kld

    ### compute Kendall's tau, Spearman's rho and Pearson correlation coefficient
    tau, _ = stats.kendalltau(learnt_values, ref_values)
    rho, _ = stats.spearmanr(learnt_values, ref_values)
    pcc, _ = stats.pearsonr(learnt_values, ref_values)
    metrics_dic['tau'] = tau
    metrics_dic['rho'] = rho
    metrics_dic['pcc'] = pcc

    ### compute nDCG
    ll = np.array(ref_values)[np.flip(np.argsort(learnt_values), 0)]

    ndcg = ndcg_at_k(ll,int(0.01*len(ll)))
    metrics_dic['ndcg_at_1%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.05*len(ll)))
    metrics_dic['ndcg_at_5%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.1*len(ll)))
    metrics_dic['ndcg_at_10%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.2*len(ll)))
    metrics_dic['ndcg_at_20%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.5*len(ll)))
    metrics_dic['ndcg_at_50%'] = ndcg
    ndcg = ndcg_at_k(ll,len(ll))
    metrics_dic['ndcg_at_all'] = ndcg

    metrics_dic['score_of_estimated_best'] = ref_values[np.argmax(learnt_values)]
    metrics_dic['score_of_true_best'] = np.max(ref_values)

    ranked_items = np.argsort(learnt_values) # smallest score first
    metrics_dic['rank_of_best'] = float(len(learnt_values) - np.argwhere(ranked_items == np.argmax(ref_values)).flatten()[0])

    if top_answer is not None:
        #accuracy in matching the top items -- when averaged across topics, it will become an accuracy score
        metrics_dic['accuracy'] = 100 * float(top_answer == np.argmax(learnt_values))

    return metrics_dic


def evaluateSummary(cand,model,len=100):
    rouge_scorer = Rouge(ROUGE_DIR,BASE_DIR,True)
    r1, r2, rl, rsu4 = rouge_scorer(cand,[model],len)
    rouge_scorer.clean()
    dic = OrderedDict()
    dic['ROUGE-1'] = r1
    dic['ROUGE-2'] = r2
    dic['ROUGE-L'] = rl
    dic['ROUGE-SU4'] = rsu4
    dic['WeightedSum'] = 3.33*(r1/.47 + r2/.212 + rsu4/.185)
    return dic

def getMetricCorrelation(metric_scores,results):
    correlations = {}

    for metric in metric_scores:
        correlations[metric] = stats.pearsonr(metric_scores[metric],results)[0]

    return correlations

def estimateTemperature(gaps,ratios):
    initial_t = 3.
    step = 0.001
    valid_num = 0
    target = 0.
    for rr in ratios:
        if not np.isnan(rr):
            valid_num += 1
            target += rr
    target /= valid_num
    if target < 0.5:
        return len(ratios)*[float('nan')], float('nan')
    round = 0
    direction = 0
    temp = initial_t

    while True:
        round += 1
        #print('\n---round {}, temperature {}---'.format(round,temp))
        prob_list = []
        for gap in gaps:
            prob_list.append(sigmoid(gap,temp))
        predict = np.mean(prob_list)
        #print('predict {}, target {}'.format(predict,target))

        if np.abs(predict-target) < 0.0001:
            break

        if predict < target:
            if direction == 1:
                break
            else:
                temp -= step
                direction = -1
        else:
            if direction == -1:
                break
            else:
                temp += step
                direction = 1

    return prob_list, temp


def plotAgreement(truer, learntr, bin_num=20, plot=True, sample_num=5000):
    assert len(truer) == len(learntr)
    rnd_idx = np.random.permutation(len(truer))[:sample_num]
    true_rewards = list(truer[rnd_idx])
    learnt_rewards = list(learntr[rnd_idx])

    agreement = [0]*bin_num
    total_case = [0]*bin_num
    gaps = np.linspace(min(true_rewards),max(true_rewards),bin_num)
    bin_width = (max(true_rewards)-min(true_rewards))/bin_num

    for i in range(0,len(true_rewards)-1):
        for j in range(i+1,len(true_rewards)):
            true_gap = true_rewards[i]-true_rewards[j]
            learnt_gap = learnt_rewards[i]-learnt_rewards[j]
            label = 0
            if true_gap*learnt_gap >= 0:
                label = 1
            idx = int(np.abs(true_gap)/bin_width)
            if idx >= bin_num:
                idx = bin_num-1

            agreement[idx] += label
            total_case[idx] += 1.

    agreement_ratio = []
    for ii in range(len(agreement)):
        if total_case[ii]>=1:
            agreement_ratio.append(agreement[ii]/total_case[ii])
        else:
            agreement_ratio.append(float('nan'))

    probs, temp = estimateTemperature(gaps,agreement_ratio)

    cross_entropy = np.zeros(bin_num)

    for i in range(0,len(true_rewards)-1):
        for j in range(i+1,len(true_rewards)):

            true_gap = true_rewards[i]-true_rewards[j]
            true_label = true_gap > 0

            learnt_gap = learnt_rewards[i]-learnt_rewards[j]
            prob =  1.0 / (1.0 + np.exp(-learnt_gap / temp))

            if prob == 1.0:
                prob -= 1e-10
            elif prob == 0:
                prob += 1e-10

            idx = int(np.abs(true_gap)/bin_width)
            if idx >= bin_num:
                idx = bin_num-1

            cross_entropy[idx] -= true_label * np.log(prob) + (1 - true_label) * np.log(1 - prob)

    # the mean cross entropy in each true gap size bin
    total_case = np.array(total_case).astype(int)
    norm_cross_entropy = np.array(cross_entropy[total_case > 0]) / total_case[total_case > 0]

    mse = 0.
    valid_cnt = 0
    for ii in range(len(agreement_ratio)):
        if not np.isnan(agreement_ratio[ii]):
            mse += np.power(agreement_ratio[ii]-probs[ii],2)
            valid_cnt += 1
    mse /= valid_cnt
    #print('temperature {}, mse {}, rmse {}'.format(temp,mse,np.sqrt(mse)))

    if plot == True:
        plt.plot(gaps,agreement_ratio,label='Observed')
        if probs is not None:
            plt.plot(gaps,probs,label='LNO regression')
        plt.legend()
        plt.title('temp {}, rmse {}'.format(temp,np.sqrt(mse)))
        plt.show()
    elif plot is not False:
        plt.plot(gaps,agreement_ratio,label='Observed')
        if probs is not None:
            plt.plot(gaps,probs,label='LNO regression')
        plt.legend()
        plt.title('temp {}, rmse {}'.format(temp,np.sqrt(mse)))
        plt.savefig(plot)

    mse = 0.
    for i in range(len(agreement_ratio)):
        if probs is not None and not np.isnan(agreement_ratio[i]) and not np.isnan(probs[i]):
            mse += np.power(agreement_ratio[i]-probs[i],2)
    rmse = np.sqrt(mse/len(probs))
    return rmse, temp, np.mean(norm_cross_entropy)


