import argparse
import time

from gensim.models import KeyedVectors
from elasticsearch import Elasticsearch

from swem import MeCabTokenizer
from swem import SWEM

# args
parser = argparse.ArgumentParser()
parser.add_argument('--word_vectors_file', type=str, required=True,
    help='Word vectors file (.txt)')
args = parser.parse_args()

client = Elasticsearch()
SEARCH_SIZE = 10
INDEX_NAME = "wikipedia"

w2v = KeyedVectors.load_word2vec_format(args.word_vectors_file, binary=False)
tokenizer = MeCabTokenizer("-O wakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
swem = SWEM(w2v, tokenizer)


def run_query_loop():
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            return


def handle_query():
    query = input("Enter query text: ")

    embedding_start = time.time()
    query_vector = swem.average_pooling(query).tolist()
    embedding_time = time.time() - embedding_start

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    search_start = time.time()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title", "text"]}
        }
    )
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"]["title"])
        print(hit["_source"]["text"][:200])
        print()


if __name__ == "__main__":
    run_query_loop()
