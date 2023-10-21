"""Function printing python version."""
import pandas as pd
import pre_process
import training_model

from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine



def load_google_vm():
    """Function printing python version."""
    wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                        binary=True,
                                        limit=100000)
    wv.save_word2vec_format('models/vectors.csv')
    return wv

def prepare_training_model():
    """Function printing python version."""
    df = pd.read_csv('phrases.csv', encoding='unicode_escape')

    w2v_df = pre_process.clean_process(df, "Phrases")

    w2v_model = training_model.train_w2v(w2v_df)
    return w2v_model


def get_phrase_vector(phrase, model):
    """Function printing python version."""
    tokens = phrase.lower().split()
    phrase_vector = [model[word] for word in tokens if word in model]
    if not phrase_vector:
        return None
    return sum(phrase_vector) / len(phrase_vector)

def main():
    """Function printing python version."""
    w2v_model = load_google_vm()

    phrase1 = input('First word:\n')
    phrase2 = input('Second word:\n')
    vector1 = get_phrase_vector(phrase1, w2v_model)
    vector2 = get_phrase_vector(phrase2, w2v_model)

    if vector1 is not None and vector2 is not None:
        similarity = 1 - cosine(vector1, vector2)
        print(f"Semantic similarity between '{phrase1}' and '{phrase2}': '{similarity}'")
    else:
        print("One or both phrases contain words not in the model")

if __name__ == "__main__":
    main()