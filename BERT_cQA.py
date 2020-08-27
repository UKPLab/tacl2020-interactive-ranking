"""
Train BERT-based models for each of the three cQA StackExchange topics: Apple, Cooking and Travel. Make predictions
for the corresponding test sets and save to a CSV file.

Steps for each topic:
1. Load training data.
2. Load validation data.
3. Initialise and train a model. <SKIP>
4. Sanity check by testing on the training and validation sets. <SKIP>
5. Load test data.
6. Compute predictions. <INITIALLY put 1 for the gold and 0 for the other answers>
7. Write to CSV files.

Each topic stores CSV files under 'data/BERT_cQA_vec_pred/%s/' % topic.
For each question in the test dataset, we save a separate CSV file.
The csv files contain a row per candidate answer + a row at the end for the gold answer (this row will be a
 duplicate of one of the others.)
The columns are: 'answer' (the text of the answer), 'prediction' (the score), 'vector' (the embedding of the answer
taken from our final BERT layer).

"""
import csv
import sys

import pandas as pd
import logging
import os
import numpy as np
import torch
from torch.nn import ReLU
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, DistilBertModel, \
    DistilBertTokenizer
from torch import nn, tensor, dtype

logging.basicConfig(level=logging.INFO)


class BertRanker(nn.Module):
    def __init__(self):
        super(BertRanker, self).__init__()
        bert = BertModel.from_pretrained("bert-base-cased")
        self.embedding_size = bert.config.hidden_size

        # self.bert = DistilBertModel.from_pretrained("distilbert-base-cased")
        self.bert = nn.DataParallel(bert)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.pooling = nn.DataParallel(self.pooling)

        # self.out = nn.Linear(bert.config.hidden_size, 1)
        self.W1 = nn.Linear(self.embedding_size, 100)
        self.W1 = nn.DataParallel(self.W1)

        self.W2 = nn.Linear(100, 10)
        self.W2 = nn.DataParallel(self.W2)

        self.out = nn.Linear(10, 1)  # only need one output because we just want a rank score

        self.relu = ReLU()

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        sequence_emb = self.bert(
            input_ids=input_ids1,
            attention_mask=attention_mask1
        )[0]
        sequence_emb = sequence_emb.transpose(1, 2)
        pooled_output_1 = self.pooling(sequence_emb)
        pooled_output_1 = pooled_output_1.transpose(2, 1)

        h1_1 = self.relu(self.W1(pooled_output_1))
        h2_1 = self.relu(self.W2(h1_1))
        scores_1 = self.out(h2_1)

        sequence_emb = self.bert(
            input_ids=input_ids2,
            attention_mask=attention_mask2
        )[0]
        sequence_emb = sequence_emb.transpose(1, 2)
        pooled_output_2 = self.pooling(sequence_emb)
        pooled_output_2 = pooled_output_2.transpose(2, 1)

        h1_2 = self.relu(self.W1(pooled_output_2))
        h2_2 = self.relu(self.W2(h1_2))
        scores_2 = self.out(h2_2)

        return scores_1, scores_2

    def forward_single_item(self, input_ids, attention_mask):
        sequence_emb = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        sequence_emb = sequence_emb.transpose(1, 2)
        pooled_output = self.pooling(sequence_emb)
        pooled_output = pooled_output.transpose(2, 1)

        h1 = self.relu(self.W1(pooled_output))
        h2 = self.relu(self.W2(h1))
        scores = self.out(h2)

        return scores, torch.squeeze(pooled_output).detach()


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()

    losses = []
    ncorrect = 0
    count_examples = 0

    for step, batch in enumerate(data_loader):
        if np.mod(step, 100) == 0:
            print("Training step %i / %i" % (step, len(data_loader)))

        input_ids1 = batch["input_ids1"].to(device)
        attention_mask1 = batch["attention_mask1"].to(device)

        input_ids2 = batch["input_ids2"].to(device)
        attention_mask2 = batch["attention_mask2"].to(device)

        scores_1, scores_2 = model(
            input_ids1=input_ids1,
            attention_mask1=attention_mask1,
            input_ids2=input_ids2,
            attention_mask2=attention_mask2
        )

        ncorrect += float(torch.sum(torch.gt(scores_1, scores_2)))  # first score is always meant to be higher
        count_examples += len(scores_1)

        loss = loss_fn(scores_1, scores_2, batch['targets'].to(device))
        losses.append(float(loss.item()))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return ncorrect / float(count_examples), np.mean(losses)


def train_bertcqa(data_loader, nepochs=1, random_seed=42, save_path='saved_bertcqa_params', reload_model=False):

    # For reproducibility while debugging. TODO: vary this during real experiments.
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Get the device for running the training and prediction
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Selecting device -- using cuda')
        print('Selected device: ')
        print(device)
        print('Current cuda device:')
        print(torch.cuda.current_device())
    else:
        device = torch.device("cpu")
        print('Selecting device -- using CPU')

    # Create the BERT-based model
    model = BertRanker()
    model = model.to(device)

    if reload_model and os.path.exists(save_path+'.pkl'):
        print('Found a previously-saved model... reloading')
        model.load_state_dict(torch.load(save_path+'.pkl'))
        with open(save_path+'_num_epochs.txt', 'r') as fh:
            epochs_completed = int(fh.read())
            print('Number of epochs already completed: %i' % epochs_completed)
    else:
        epochs_completed = 0

    optimizer = AdamW(model.parameters(), lr=5e-5, correct_bias=False)
    optimizer.zero_grad()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(data_loader) * nepochs * 0.1,
        num_training_steps=len(data_loader) * nepochs
    )

    loss_fn = nn.MarginRankingLoss(margin=0.0).to(device)

    for epoch in range(epochs_completed, nepochs):
        print('Training epoch %i' % epoch)
        train_acc, train_loss = train_epoch(
            model,
            data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler
        )
        print(f'Train loss {train_loss} pairwise label accuracy {train_acc}')

        print('Saving trained model')
        torch.save(model.state_dict(), save_path+'.pkl')
        # write the number of epochs to file. If we need to restart training, we don't need to repeat all epochs.
        with open(save_path+'_num_epochs.txt', 'w') as fh:
            fh.write(str(epoch+1))

    return model, device


def predict_bertcqa(model, data_loader, device):
    scores = np.zeros(0)
    vectors = np.zeros((0, model.embedding_size))
    qids = np.zeros(0)
    ismatch = np.zeros(0)

    model.eval()

    for step, batch in enumerate(data_loader):
        if np.mod(step, 100) == 0:
            print("Prediction step  %i / %i" % (step, len(data_loader)))

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        batch_scores, batch_vectors = model.forward_single_item(input_ids, attention_mask)

        print('step %i' % step)
        print('batch_vctor shape ' + str(batch_vectors.shape))
        print('vectors shape ' + str(vectors.shape))

        scores = np.append(scores, batch_scores.cpu().detach().numpy().flatten())
        batch_vectors = batch_vectors.cpu().numpy()
        if batch_vectors.ndim == 1:
            batch_vectors = batch_vectors[None, :]
        vectors = np.concatenate((vectors, batch_vectors), axis=0)
        qids = np.append(qids, batch["qid"].detach().numpy().flatten())
        ismatch = np.append(ismatch, batch["ismatch"].detach().numpy().flatten())

    print('Outputting an embedding vector with shape ' + str(np.array(vectors).shape))

    return scores, vectors, qids, ismatch


def evaluate_accuracy(model, data_loader, device):
    scores, vectors, qids, matches = predict_bertcqa(model, data_loader, device)

    unique_questions = np.unique(qids)

    ncorrect = 0
    nqs = len(unique_questions)

    for q_id in unique_questions:
        qscores = scores[qids == q_id]
        isgold = matches[qids == q_id]

        if isgold[np.argmax(qscores)]:
            ncorrect += 1

    acc = ncorrect / float(nqs)
    print("Accuracy = %f" % acc)
    return acc, scores, vectors


# Create the dataset class
class SEPairwiseDataset(Dataset):
    def __init__(self, qa_pairs: list):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        # BertTokenizer.from_pretrained('bert-base-cased')
        self.qa_pairs = qa_pairs
        self.max_len = 512

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, i):
        # first item in the pair
        encoding1 = self.tokenizer.encode_plus(
            self.qa_pairs[i][0],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # second item in the pair
        encoding2 = self.tokenizer.encode_plus(
            self.qa_pairs[i][1],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text1': self.qa_pairs[i][0],
            'text2': self.qa_pairs[i][1],
            'input_ids1': encoding1['input_ids'].flatten(),
            'input_ids2': encoding2['input_ids'].flatten(),
            'attention_mask1': encoding1['attention_mask'].flatten(),
            'attention_mask2': encoding2['attention_mask'].flatten(),
            'targets': tensor(1, dtype=torch.float)
        }


class SESingleDataset(Dataset):
    def __init__(self, qas: list, qids: list, aids: list, goldids: dict):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        # BertTokenizer.from_pretrained('bert-base-cased')
        self.qas = qas
        self.qids = qids
        self.aids = aids
        self.goldids = goldids
        self.max_len = 512

    def __len__(self):
        return len(self.qas)

    def __getitem__(self, i):
        # first item in the pair
        encoding1 = self.tokenizer.encode_plus(
            self.qas[i],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text1': self.qas[i],
            'input_ids': encoding1['input_ids'].flatten(),
            'attention_mask': encoding1['attention_mask'].flatten(),
            'targets': tensor(1, dtype=torch.long),
            'qid': self.qids[i],
            'ismatch': self.goldids[self.qids[i]] == self.aids[i]
        }


def construct_pairwise_dataset(dataframe, n_neg_samples=10):
    """
    Function for constructing a pairwise training set where each pair consists of a matching QA sequence and a
    non-matching QA sequence.
    :param n_neg_samples: Number of pairs to generate for each question by sampling non-matching answers and pairing
    them with matching answers.
    :param dataframe:
    :return:
    """
    # Get the positive (matching) qs and as from traindata and put into pairs
    # Sample a number of negative (non-matching) qs and as from the answers listed for each question in traindata

    qa_pairs = []

    for idx, qid in enumerate(dataframe.index):
        # Reconstruct the text sequences for the training questions
        tokids = questions.loc[qid].values[0].split(' ')
        toks = vocab[np.array(tokids).astype(int)]
        question = ' '.join(toks)

        # Reconstruct the text sequences for the true answers
        gold_ans_id = dataframe.loc[qid]["goldid"]

        # some of the lines seem to have two gold ids. Just use the first.
        gold_ans_ids = gold_ans_id.split(' ')
        gold_ans_id = gold_ans_ids[0]

        tokids = answers.loc[gold_ans_id].values[0].split(' ')
        toks = vocab[np.array(tokids).astype(int)]
        gold_ans = ' '.join(toks)

        # Join the sequences. Insert '[SEP]' between the two sequences
        qa_gold = question + ' [SEP] ' + gold_ans

        # Reconstruct the text sequences for random wrong answers
        wrong_ans_ids = dataframe.loc[qid]["ansids"]
        wrong_ans_ids = wrong_ans_ids.split(' ')
        if len(wrong_ans_ids) < n_neg_samples + 1:
            continue

        if n_neg_samples == 0:
            # use all the wrong answers (exclude the gold one that is mixed in)
            n_wrongs = len(wrong_ans_ids) - 1
            widx = 0
        else:
            # use a specified sample size
            n_wrongs = n_neg_samples

        qa_wrongs = []
        while len(qa_wrongs) < n_wrongs:
            if n_neg_samples == 0:
                # choose the next wrong answer, skip over the gold answer.
                wrong_ans_id = wrong_ans_ids[widx]
                widx += 1
                if wrong_ans_id == gold_ans_id:
                    wrong_ans_id = wrong_ans_ids[widx]
                    widx += 1
            else:
                # choose a new negative sample
                wrong_ans_id = gold_ans_id
                while wrong_ans_id == gold_ans_id:
                    wrong_ans_id = wrong_ans_ids[np.random.randint(len(wrong_ans_ids))]

            tokids = answers.loc[wrong_ans_id].values[0].split(' ')
            toks = vocab[np.array(tokids).astype(int)]
            wrong_ans = ' '.join(toks)

            qa_wrong = question + ' [SEP] ' + wrong_ans
            qa_wrongs.append(qa_wrong)
            qa_pairs.append((qa_gold, qa_wrong))

    data_loader = DataLoader(
        SEPairwiseDataset(qa_pairs),
        batch_size=16,
        num_workers=8
    )

    data = next(iter(data_loader))

    return qa_pairs, data_loader, data


def construct_single_item_dataset(dataframe):
    """
    Constructs a dataset where each element is a single QA pair. It contains all the sampled non-matching answers from
    the given dataframe. A list of question IDs is returned to indicate which items relate to the same question,
    along with a dict with question IDs as keys and the indexes of the gold answers within the answers for their
    corresponding questions as items.
    :param dataframe:
    :return:
    """
    # Get the positive (matching) qs and as from traindata and put into pairs
    # Sample a number of negative (non-matching) qs and as from the answers listed for each question in traindata

    qas = []
    qids = []
    aids = []
    goldids = {}

    for idx, qid in enumerate(dataframe.index):
        # Reconstruct the text sequences for the training questions
        tokids = questions.loc[qid].values[0].split(' ')
        toks = vocab[np.array(tokids).astype(int)]
        question = ' '.join(toks)

        # Reconstruct the text sequences for random wrong answers
        ans_ids = dataframe.loc[qid]["ansids"]
        ans_ids = ans_ids.split(' ')
        if len(ans_ids) < 2:
            continue
        ans_ids = np.unique(ans_ids) # make sure we don't have the same answer multiple times

        gold_id = dataframe.loc[qid]["goldid"]
        gold_id = gold_id.split(' ')
        gold_id = gold_id[0]

        for ans_idx, ans_id in enumerate(ans_ids):
            tokids = answers.loc[ans_id].values[0].split(' ')
            toks = vocab[np.array(tokids).astype(int)]
            wrong_ans = ' '.join(toks)

            qa_wrong = question + ' [SEP] ' + wrong_ans
            qas.append(qa_wrong)
            qids.append(qid)
            aids.append(ans_id)

            if ans_id == gold_id:
                goldids[qid] = ans_id

        if qid not in goldids:
            print("Didn't find the goldid in the list of candidates for q %i. Gold ID = %s, answers = %s" %
                  (qid, dataframe.loc[qid]["goldid"], dataframe.loc[qid]["ansids"]))

    data_loader = DataLoader(
        SESingleDataset(qas, qids, aids, goldids),
        batch_size=16,
        num_workers=8
    )

    data = next(iter(data_loader))

    return qas, qids, goldids, aids, data_loader, data


if __name__ == "__main__":

    # Create the output dir
    outputdir = './data/cqa_base_models/BERT_vec_pred'
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    # Our chosen topics
    topic = sys.argv[1]  # ['apple', 'cooking', 'travel']

    print('Loading the training data for %s' % topic)

    # Data directory:
    datadir = './data/cqa_data/%s.stackexchange.com' % topic

    # answers.tsv: each row corresponds to an answer; first column is answer ID; rows contain
    # the space-separated IDs of the tokens in the answers.
    # questions.tsv: as above but first column is question ID, rows contain space-separated token IDs of questions.
    # train.tsv, test.tsv and valid.tsv contain the questions & candidates in each set. First column is question ID,
    # second column is gold answer ID, third column is a space-separated list of candidate answer IDs.
    # vocab.tsv is needed to retrieve the text of the questions and answers from the token IDs. First col is ID and
    # second col is the token.

    # load the vocab
    vocab = pd.read_csv(os.path.join(datadir, 'vocab.tsv'), sep='\t', quoting=csv.QUOTE_NONE, header=None,
                        index_col=0, names=['tokens'], dtype=str, keep_default_na=False)["tokens"].values

    # load the questions
    questions = pd.read_csv(os.path.join(datadir, 'questions.tsv'), sep='\t', header=None, index_col=0)

    # load the answers
    answers = pd.read_csv(os.path.join(datadir, 'answers.tsv'), sep='\t', header=None, index_col=0)

    # Load the training set
    traindata = pd.read_csv(os.path.join(datadir, 'train.tsv'), sep='\t', header=None, names=['goldid', 'ansids'],
                            index_col=0)
    tr_qa_pairs, tr_data_loader, tr_data = construct_pairwise_dataset(traindata, n_neg_samples=20)

    qmax = 0
    noverlength = 0
    for q in tr_qa_pairs:
        l = len(q[0].split(' '))
        qmax = l if l > qmax else qmax
        if l > 512:
            noverlength += 1

    print('QuestionAnswer max length: %i' % qmax)
    print('Number over length = %i' % noverlength)
    print('number of qas = %i' % len(tr_qa_pairs))

    # Train the model ----------------------------------------------------------------------------------------------
    bertcqa_model, device = train_bertcqa(tr_data_loader, 3, 42, os.path.join(outputdir, 'model_params_%s' % topic),
                                          reload_model=True)

    # Compute performance on training set --------------------------------------------------------------------------
    # Training is very large, don't bother with this.
    # print("Evaluating on training set:")
    # tr_qas2, tr_qids2, tr_goldids2, tr_aids2, tr_data_loader2, tr_data2 = construct_single_item_dataset(traindata)
    # evaluate_accuracy(bertcqa_model, tr_data_loader2, device)

    # Compute performance on validation set ------------------------------------------------------------------------
    # Load the validation set
    # validationdata = pd.read_csv(os.path.join(datadir, 'valid.tsv'), sep='\t', header=None,
    #                              names=['goldid', 'ansids'], index_col=0, nrows=2)
    # va_qas, va_qids, va_goldids, va_aids, va_data_loader, va_data = construct_single_item_dataset(validationdata)
    #
    # print("Evaluating on validation set:")
    # evaluate_accuracy(bertcqa_model, va_data_loader, device, va_qids, va_goldids)

    # Compute performance on test set ------------------------------------------------------------------------------

    # Load the test set
    testdata = pd.read_csv(os.path.join(datadir, 'test.tsv'), sep='\t', header=None, names=['goldid', 'ansids'],
                           index_col=0)
    te_qas, te_qids, te_goldids, te_aids, te_data_loader, te_data = construct_single_item_dataset(testdata)

    print("Evaluating on test set:")
    _, te_scores, te_vectors = evaluate_accuracy(bertcqa_model, te_data_loader, device)

    # Output predictions in the right format for the GPPL experiments ----------------------------------------------
    # Save predictions for the test data
    fname_text = os.path.join(outputdir, '%s_text.tsv' % topic)
    fname_numerical = os.path.join(outputdir, '%s_num.tsv' % topic)

    # The text data and other info goes here:
    text_df = pd.DataFrame(columns=['qid', 'answer', 'isgold'])
    # Store the prediction and embedding vectors here:
    numerical_data = np.empty((len(te_qids), 1 + bertcqa_model.embedding_size))

    for i, qid in enumerate(te_qids):
        if np.mod(i, 100) == 0:
            print("Outputting qa pair %i / %i" % (i, len(te_qids)))

        goldid = te_goldids[qid]

        ansid = te_aids[i]
        tokids = answers.loc[ansid].values[0].split(' ')
        toks = vocab[np.array(tokids).astype(int)]
        answer_text = ' '.join(toks)

        score = te_scores[i]
        vector = te_vectors[i]

        isgold = True if goldid == ansid else False

        text_df = text_df.append(
            {'qid': qid, 'answer': answer_text, 'isgold': isgold},
            ignore_index=True
        )

        numerical_data[i, 0] = score
        numerical_data[i, 1:] = vector

    text_df.to_csv(fname_text, sep='\t')
    pd.DataFrame(numerical_data).to_csv(fname_numerical, sep='\t')
