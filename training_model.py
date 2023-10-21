"""Function printing python version."""
import multiprocessing

from gensim.models import Word2Vec


def train_w2v(df_w2v):
    """Function printing python version."""
    cores = multiprocessing.cpu_count()
    model_w2v = Word2Vec(df_w2v,
                         min_count=4,
                         window=4,
                         workers=cores-1,
                         vector_size=200)

    model_w2v.build_vocab(df_w2v, progress_per=10000)
    model_w2v.train(df_w2v, total_examples=model_w2v.corpus_count, epochs=100, report_delay=1)
    
    word_vectors = model_w2v.wv

    word_vectors.save('models/trained.model')

