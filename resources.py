import os

#BASE_DIR = os.path.abspath('.')
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #os.path.abspath('.')
ROUGE_DIR = os.path.join(BASE_DIR,'summariser','rouge','ROUGE-RELEASE-1.5.5/') #do not delete the '/' at the end
PROCESSED_PATH = os.path.join(BASE_DIR,'data','summaries_processed_data')
SUMMARY_DB_DIR = os.path.join(BASE_DIR,'data','sampled_summaries')
DOC_SEQUENCE_PATH = os.path.join(BASE_DIR,'summariser','utils','DocsSequence.txt')

LANGUAGE = 'english'
