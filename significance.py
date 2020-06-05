import json, numpy as np, pandas as pd, os
from scipy.stats import wilcoxon

topics = ['apple', 'cooking', 'travel']
metrics = ['ndcg_at_1%', 'accuracy']

for topic in topics:
    # baseline directory
    baseline = 'results_coala/lno3_lr_%s_rep0/' % topic

    # imp directory
    impdir = 'results_coala/lno03_gpplhh_%s_rep0/' % topic

    metrics_base = os.path.join(baseline, 'metrics_unc_LR_10.json')

    metrics_imp = os.path.join(impdir, 'metrics_imp_GPPLHH_10.json')

    with open (metrics_base, 'r') as fh:
        metrics_base = json.load(fh)

    with open (metrics_imp, 'r') as fh:
        metrics_imp = json.load(fh)

    for metric in metrics:
        mbase = metrics_base[metric]
        mimp = metrics_imp[metric]

        _, p = wilcoxon(mbase, mimp)

        print('For topic %s, %s %s significantly different with p = %f' % (topic, metric, 'is' if p<0.05 else 'ISNT', p))


### store all results
all_result_dic = {}
topic_cnt = 0

datasets = ['duc01', 'duc02', 'duc04']
metrics = ['ndcg_at_1%', 'pcc']

for dataset in datasets:
    # baseline directory
    baseline = 'results/%s_lr_10inter_rep0/' % dataset

    # imp directory
    impdir = 'results/%s_gpplhh_10inter_rep0/' % dataset

    metrics_base = os.path.join(baseline, 'metrics_unc_LR_10.json')

    metrics_imp = os.path.join(impdir, 'metrics_imp_GPPLHH_10.json')

    with open (metrics_base, 'r') as fh:
        metrics_base = json.load(fh)

    with open (metrics_imp, 'r') as fh:
        metrics_imp = json.load(fh)

    for metric in metrics:
        mbase = metrics_base[metric]
        mimp = metrics_imp[metric]

        _, p = wilcoxon(mbase, mimp)

        print('For dataset %s, %s %s significantly different with p = %f' % (dataset, metric, 'is' if p<0.05 else 'ISNT', p))

for dataset in datasets:
    # baseline directory
    baseline = 'results/%s_lr_100inter_rep0/' % dataset

    # imp directory
    impdir = 'results/%s_gpplhh_100inter_rep0/' % dataset

    metrics_base = os.path.join(baseline, 'metrics_unc_LR.json')

    metrics_imp = os.path.join(impdir, 'metrics_imp_GPPLHH.json')

    with open (metrics_base, 'r') as fh:
        metrics_base = json.load(fh)

    with open (metrics_imp, 'r') as fh:
        metrics_imp = json.load(fh)

    for metric in metrics:
        mbase = metrics_base[metric]
        mimp = metrics_imp[metric]

        _, p = wilcoxon(mbase, mimp)

        print('For dataset %s, %s %s significantly different with p = %f' % (dataset, metric, 'is' if p<0.05 else 'ISNT', p))