transformers, Ranx, pyterrier, and pandas are required to run both
'query-expansion.py' and 'reranking.py'.
Both files assume all the project files (topic & answer files) are in a folder 
named 'materials' (see github for an example).
In addition, 'common.py' must be in the same directory as well to run.

Both files take zero arguments, and produce four result files each.
(e.g. 'qe_p1_q1.tsv' => query expansion system, prompt 1, topics_1 queries).

'trec_eval' folder contains evaluation results from trec_eval.
