from bert_serving.server.bert.extract_features import convert_lst_to_features
from bert_serving.server.bert.tokenization import FullTokenizer
from tensorflow.keras.utils import Progbar
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.estimator import Estimator
import pandas as pd
import numpy as np
import logging
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pickle import dump

from bert_serving.server.graph import optimize_graph
from bert_serving.server.helper import get_args_parser

articles = pd.read_csv('data/EnglishArticles.csv')
articles = articles.drop([['cleaned_text']])
print(articles.shape)
print(articles.head())
print(articles.isnull().sum())

bert_train, bert_test = train_test_split(
    articles[['TITLE', 'cleaned_sumary', 'bert_cleaned_text']], test_size=0.33, random_state=33)
bert_summaries_train_array = bert_train.values
bert_summaries_test_array = bert_test.values


MODEL_DIR = '../content/bert/'  # @param {type:"string"}
GRAPH_DIR = '../content/graph/'  # @param {type:"string"}
GRAPH_OUT = 'extractor.pbtxt'  # @param {type:"string"}
GPU_MFRAC = 0.2  # @param {type:"string"}

POOL_STRAT = 'REDUCE_MEAN'  # @param {type:"string"}
POOL_LAYER = "-2"  # @param {type:"string"}
SEQ_LEN = "512"  # @param {type:"string"}

tf.gfile.MkDir(GRAPH_DIR)

parser = get_args_parser()
carg = parser.parse_args(args=['-model_dir', MODEL_DIR,
                               "-graph_tmp_dir", GRAPH_DIR,
                               '-max_seq_len', str(SEQ_LEN),
                               '-pooling_layer', str(POOL_LAYER),
                               '-pooling_strategy', POOL_STRAT,
                               '-gpu_memory_fraction', str(GPU_MFRAC)])

tmpfi_name, config = optimize_graph(carg)
graph_fout = os.path.join(GRAPH_DIR, GRAPH_OUT)

tf.gfile.Rename(
    tmpfi_name,
    graph_fout,
    overwrite=True
)
print("Serialized graph to {}".format(graph_fout))


log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
log.handlers = []


GRAPH_PATH = "../content/graph/extractor.pbtxt"  # @param {type:"string"}
VOCAB_PATH = "../content/bert/vocab.txt"  # @param {type:"string"}

SEQ_LEN = 512  # @param {type:"integer"}

INPUT_NAMES = ['input_ids', 'input_mask', 'input_type_ids']
bert_tokenizer = FullTokenizer(VOCAB_PATH)


def build_feed_dict(texts):
    text_features = list(convert_lst_to_features(
        texts, SEQ_LEN, SEQ_LEN,
        bert_tokenizer, log, False, False))

    target_shape = (len(texts), -1)

    feed_dict = {}
    for iname in INPUT_NAMES:
        features_i = np.array([getattr(f, iname) for f in text_features])
        features_i = features_i.reshape(target_shape)
        features_i = features_i.astype("int32")
        feed_dict[iname] = features_i

    return feed_dict


def build_input_fn(container):

    def gen():
        while True:
            try:
                yield build_feed_dict(container.get())
            except:
                yield build_feed_dict(container.get())

    def input_fn():
        return tf.data.Dataset.from_generator(
            gen,
            output_types={iname: tf.int32 for iname in INPUT_NAMES},
            output_shapes={iname: (None, None) for iname in INPUT_NAMES})
    return input_fn


class DataContainer:
    def __init__(self):
        self._texts = None

    def set(self, texts):
        if type(texts) is str:
            texts = [texts]
        self._texts = texts

    def get(self):
        return self._texts


def model_fn(features, mode):
    with tf.gfile.GFile(GRAPH_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    output = tf.import_graph_def(graph_def,
                                 input_map={k + ':0': features[k]
                                            for k in INPUT_NAMES},
                                 return_elements=['final_encodes:0'])

    return EstimatorSpec(mode=mode, predictions={'output': output[0]})


estimator = Estimator(model_fn=model_fn)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def build_vectorizer(_estimator, _input_fn_builder, batch_size=128):
    container = DataContainer()
    predict_fn = _estimator.predict(_input_fn_builder(
        container), yield_single_examples=False)

    def vectorize(text, verbose=False):
        x = []
        bar = Progbar(len(text))
        for text_batch in batch(text, batch_size):
            container.set(text_batch)
            x.append(next(predict_fn)['output'])
            if verbose:
                bar.add(len(text_batch))
        r = np.vstack(x)
        return r
    return vectorize


bert_vectorizer = build_vectorizer(estimator, build_input_fn)

X_train, X_test, names = [], [], []

for article_title, article_summary in bert_summaries_train_array:
    X_train.append(article_summary)
    names.append(article_title)

for article_title, article_summary in bert_summaries_test_array:
    X_test.append(article_summary)
    names.append(article_title)

dump(names, open('data/names.txt'), 'wb')

summary_train_vector = bert_vectorizer(X_train, verbose=True)
print('Train set vectorized.')
np.save('bert/train_vector', summary_train_vector)

summary_test_vector = bert_vectorizer(X_test, verbose=True)
print('Test set vectorized.')
np.save('bert/test_vector', summary_test_vector)
