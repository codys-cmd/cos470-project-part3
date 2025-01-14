import os, re, csv, json, math, torch

#Preprocessing functions
htmlRegex = re.compile('<[^>?]+>')
htmlSplit = lambda t: ' '.join([st for st in htmlRegex.split(t)])

#Loading functions
def read_qrel_file(qrelPath):
    result = {}
    reader = csv.reader(open(qrelPath, mode='r', encoding='utf-8'),
                        delimiter='\t', lineterminator='\n')
    for line in reader:
        qid = line[0]
        docId = line[2]
        score = int(line[3])
        if qid in result:
            result[qid][docId] = score
        else:
            result[qid] = {docId: score}
            
    return result

def read_topic_file(topicPath):
    result = {}
    queries = json.load(open(topicPath, mode='r', encoding='utf-8'))
    for query in queries:
        title = htmlSplit(query['Title'])
        body = htmlSplit(query['Body'])
        result[query['Id']] = {'title': title,
                               'body': body,
                               'text': ''.join([title, ' ', body]),
                               'tags': query['Tags']}
    return result

def read_collection_file(collectionPath):
    result = {}
    collection = json.load(open(collectionPath, mode='r', encoding='utf-8'))
    for doc in collection:
        result[doc['Id']] = htmlSplit(doc['Text'])
    return result
