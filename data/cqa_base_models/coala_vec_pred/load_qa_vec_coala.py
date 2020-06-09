import pickle

def load_qa_vec_coala(fname):
    qa_list, vec_list, pred_list = pickle.load( open(fname,'rb'), encoding='latin1' )

    print('sanity check')
    assert len(qa_list) == len(vec_list) == len(pred_list)
    print('{} questions in total'.format(len(qa_list)))
    for ii in range(len(qa_list)):
        assert len(qa_list[ii]['pooled_answers']) == len(vec_list[ii])-1 == len(pred_list[ii])-1
        # the last item in vec_list[ii] is the vector for gold_answer. The same applies to pred_list. this explains why the length of vec_list and pred_list is one plus len(pooled_answer)
        print('question',ii,'candidate answer num: ',len(qa_list[ii]['pooled_answers']))

if __name__ == '__main__':
    fname = 'qa_vec_coala/se_cooking_coala.qa_vec_pred'
    load_qa_vec_coala(fname)
        
        

