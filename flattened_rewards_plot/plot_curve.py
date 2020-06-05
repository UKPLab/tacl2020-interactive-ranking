import matplotlib.pyplot as plt

def readRouge(fpath, rouge_n):
    pct_list = []
    rouge_list = []

    with open(fpath,'r') as ff:
        for line in ff.readlines():
            if 'cut percentage' in line:
                pct_list.append(float(line.split(':')[1]))
                continue

            if 'ROUGE-{}'.format(rouge_n) in line:
                rouge_list.append(float(line.split(':')[1]))

    return pct_list, rouge_list

if __name__ == '__main__':
    #wanted_rouge = '2' # 1, 2, L, SU4
    pct_list, rouge2_list = readRouge('./cut_rate.txt','2')
    pct_list, rougeL_list = readRouge('./cut_rate.txt','L')

    fsize = 13
    fig, ax1 = plt.subplots()
    ax1.plot(pct_list,rouge2_list,'b--')
    ax1.set_xlabel('Bottom Reward Percentage',fontsize=fsize)
    ax1.set_ylabel('ROUGE-{}'.format(2),color='b',fontsize=fsize)
    ax1.tick_params('y',colors='b',size=fsize,labelsize=fsize)

    ax2 = ax1.twinx()
    ax2.plot(pct_list,rougeL_list,'r-')
    ax2.set_ylabel('ROUGE-{}'.format('L'),color='r',fontsize=fsize)
    ax2.tick_params('y',colors='r',size=fsize,labelsize=fsize)

    fig.tight_layout()
    plt.grid()
    plt.show()
