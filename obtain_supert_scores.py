import os

from resources import PROCESSED_PATH, SUMMARY_DB_DIR
from ref_free_metrics.sbert_score_metrics import get_sbert_score_metrics
from summariser.utils.corpus_reader import CorpusReader
from summariser.utils.reader import readSummaries
from summariser.vector.vector_generator import Vectoriser, State
import numpy as np


class SupertVectoriser(Vectoriser):
    def getSummaryVectors(self, all_summaries_acts, use_coverage_feats=False):
        vector_list = []
        supert_scores = []
        summ_list = []

        for act_list in all_summaries_acts:
            # For each summary, we create a state object
            state = State(self.sum_token_length, self.base_length, len(self.sentences), self.block_num, self.language)

            # Now, we construct the text of the summary from the list of actions, which are IDs of sentences to add.
            for _, act in enumerate(act_list):
                state.updateState(act, self.sentences, read=True)

            if use_coverage_feats:
                # Get the original heuristic feature vectors containing the coverage/redundancy features
                vector = state.getSelfVector(self.top_ngrams_list, self.sentences)

                # Get the relevant features
                vector = vector[-5:]
            else:
                vector = []

            vector_list.append(vector)

            # Retrieve the text of the summary
            summ_text = ' '.join(state.draft_summary_list)
            #print(summ_text)
            summ_list.append(summ_text)

        # Obtain the SUPERT scores for the summaries from their embedding vectors
        supert_scores, supert_vectors = get_sbert_score_metrics(docs, summ_list, 'top15', return_summary_vectors=True)

        for i, supert_vector in enumerate(supert_vectors):
            vector_list[i] = np.append(np.mean(supert_vector, axis=0), vector_list[i])

        return vector_list, supert_scores


if __name__ == '__main__':
    for dataset in ['DUC2001', 'DUC2002', 'DUC2004']:

        # read documents and ref. summaries
        reader = CorpusReader(PROCESSED_PATH)
        data = reader.get_data(dataset)

        for topic, docs, models in data:
            summaries_acts_list, _ = readSummaries(dataset, topic, 'heuristic')
            print('num of summaries read: {}'.format(len(summaries_acts_list)))

            # Use the vectoriser to obtain summary embedding vectors
            vec = SupertVectoriser(docs)
            summary_vectors, scores = vec.getSummaryVectors(summaries_acts_list)

            # Save the vectors to the cache file
            if not os.path.exists('./data/supert'):
                os.mkdir('./data/supert')
            summary_vecs_cache_file = './data/summary_vectors/supert/summary_vectors_%s_%s.csv' % (dataset, topic)
            np.savetxt(summary_vecs_cache_file, summary_vectors)

            # Write to the output file
            output_file = os.path.join(SUMMARY_DB_DIR, dataset, topic, 'supert')
            with open(output_file, 'w') as ofh:
                for i, summ in enumerate(summaries_acts_list):
                    act_str = np.array(summ).astype(str)
                    actions_line = "actions:" + ",".join(act_str) + "\n"
                    ofh.write(actions_line)

                    utility_line = "utility:" + str(scores[i]) + "\n"
                    ofh.write(utility_line)
