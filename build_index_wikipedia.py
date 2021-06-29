import argparse
import json
import gzip

from gensim.models import KeyedVectors
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from joblib import Parallel, delayed

from swem import MeCabTokenizer
from swem import SWEM


def index_batch(docs):
    #requests = Parallel(n_jobs=-1)([delayed(get_request)(doc) for doc in docs])
    requests = []
    for doc in docs:
        requests.append(get_request(doc))
    bulk(client, requests)


def get_request(doc):
    return {"_op_type": "index",
            "_index": INDEX_NAME,
            "text": doc["text"],
            "title": doc["title"],
            "text_vector": swem.average_pooling(doc["text"]).tolist()
            }


# args
parser = argparse.ArgumentParser()
parser.add_argument('--cirrus_file', type=str, required=True,
    help='Wikipedia Cirrussearch content dump file (.json.gz)')
parser.add_argument('--word_vectors_file', type=str, required=True,
    help='Word vectors file (.txt)')
args = parser.parse_args()

# embedding
w2v = KeyedVectors.load_word2vec_format(args.word_vectors_file, binary=False)
tokenizer = MeCabTokenizer("-O wakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
swem = SWEM(w2v, tokenizer)

# elasticsearch
client = Elasticsearch()
BATCH_SIZE = 1000
INDEX_NAME = "wikipedia"

# clean elastic
client.indices.delete(index=INDEX_NAME, ignore=[404])
with open("index.json") as index_file:
    source = index_file.read().strip()
    client.indices.create(index=INDEX_NAME, body=source)


# count total
total = 0
with gzip.open(args.cirrus_file) as f:
    for line in f:
        json_line = json.loads(line)
        if "index" not in json_line:
            total += 1

# build
docs = []
count = 0
with gzip.open(args.cirrus_file) as f:
    for line in f:
        json_line = json.loads(line)
        if "index" not in json_line:
            doc = json_line

            docs.append(doc)
            count += 1

            if count % BATCH_SIZE == 0:
                index_batch(docs)
                docs = []
                print(f"Indexed {count} documents. {100.0*count/total}%")
    if docs:
        index_batch(docs)
        print("Indexed {} documents.".format(count))
