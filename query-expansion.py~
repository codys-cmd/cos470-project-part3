from common import *
import sys, string, time

import pyterrier as pt
from ranx import Run, fuse, Qrels, evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from pandas import DataFrame

model_id = 'Qwen/Qwen2.5-3B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(model_id)

llmPrompt1 = ('Given the following query, provide a comma seperated '
              'list of keywords that are relevant to the query. Provide '
              'at least ten keywords.')

llmMessages1 = [
    {'role': 'system', 'content': llmPrompt1},
    {'role': 'user', 'content': ''}
]

llmPrompt2 = ('Given the following query, answer it with a single paragraph.')

llmMessages2 = [
    {'role': 'system', 'content': llmPrompt2},
    {'role': 'user', 'content': ''}
]

ptMeta = {'docno' : 10, 'text' : 10000}
ptFields = ['text']

print_time = lambda s: print(time.time() - s, flush=True)

stripPunc = lambda t: t.translate(str.maketrans('','',string.punctuation))

def create_pyterrier_query_df(qids, queries):
    getText = lambda q: stripPunc(q).lower()
    return DataFrame([[qids[i], getText(queries[i])] for i in range(len(qids))],
                     columns=['qid', 'query'])

def create_llm_text(queries, messages):
    newQueries = []
    for q in queries:
        messages[-1]['content'] = q
        inputText = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            temperature=0.5,
            add_generation_prompt=True
        )
        encodedInputs = tokenizer(inputText,
                                  return_tensors='pt').to('cuda')
        outputs = model.generate(**encodedInputs,
                                 max_new_tokens=512)
        newText = tokenizer.batch_decode(
            outputs[:, encodedInputs.input_ids.shape[1]:])[0]
        newQueries.append(newText)
        #print(newText)
        
    return newQueries

def create_ranx_run(qids, docids, searchResults, name):
    runDict = {}
    for i in range(len(qids)):
        qid = qids[i]
        runDict[qid] = {}
        for r in searchResults[i]:
            runDict[qid][docids[r['corpus_id']]] = r['score']

    run = Run(runDict, name=name)
    return run

queries = lambda q: {'text': [q['text'] for qid, q in q.items()],
                     'qids': [qid for qid, q in q.items()]}

def main():
    q1 = queries(read_topic_file('materials/topics_1.json'))
    q2 = queries(read_topic_file('materials/topics_2.json'))

    collection = read_collection_file('materials/Answers.json')
    colText = [doc for docid, doc in collection.items()]
    docids = [docid for docid, doc in collection.items()]

    if not os.path.exists('./index'):
        indexer = pt.IterDictIndexer('./index', meta=ptMeta, overwrite=True)
        ptDocs = [{'docno': docid,
                   'text': stripPunc(htmlSplit(text))}
                  for docid, text in collection.items()]
        indexer.index(ptDocs)
    
    index = pt.IndexFactory.of('./index')
    bm25 = pt.terrier.Retriever(index, wmodel='BM25', num_results=100)

    start = time.time()
    
    # Query Expansion Runs
    q1ExpandedText1 = [' '.join([z[0], z[1]])
                       for z in
                       zip(q1['text'], create_llm_text(q1['text'], llmMessages1))]
    q1Df1 = create_pyterrier_query_df(q1['qids'], q1ExpandedText1)
    rQ1Df1 = bm25.transform(q1Df1)
    print_time(start)

    start = time.time()
    
    q2ExpandedText1 = [' '.join([z[0], z[1]])
                       for z in
                       zip(q2['text'], create_llm_text(q2['text'], llmMessages1))]
    q2Df1 = create_pyterrier_query_df(q2['qids'], q2ExpandedText1)
    rQ2Df1 = bm25.transform(q2Df1)
    print_time(start)

    start = time.time()
    
    q1Df2 = create_pyterrier_query_df(q1['qids'], create_llm_text(q1['text'], llmMessages2))
    rQ1Df2 = bm25.transform(q1Df2)
    print_time(start)

    start = time.time()
    
    q2Df2 = create_pyterrier_query_df(q2['qids'], create_llm_text(q2['text'], llmMessages2))
    rQ2Df2 = bm25.transform(q2Df2)
    print_time(start)
    
    Run.from_df(rQ1Df1,
                q_id_col='qid',
                doc_id_col='docno').save('qe_p1_q1.tsv', kind='trec')
    
    Run.from_df(rQ2Df1,
                q_id_col='qid',
                doc_id_col='docno').save('qe_p1_q2.tsv', kind='trec')

    Run.from_df(rQ1Df2,
                q_id_col='qid',
                doc_id_col='docno').save('qe_p2_q1.tsv', kind='trec')
    
    Run.from_df(rQ2Df2,
                q_id_col='qid',
                doc_id_col='docno').save('qe_p2_q2.tsv', kind='trec')
    
if __name__ == '__main__':
    main()
