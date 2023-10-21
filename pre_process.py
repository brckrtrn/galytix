
"""Function printing python version."""
import pandas as pd
import re
from nltk.corpus import stopwords

def clean_process(df, target_column):
    """Function printing python version."""

    def clean_data(text):
        """Function printing python version."""
        text = re.sub(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+', '', text)
        text = re.sub(r'[\\/×\^\]\[÷]', '', text)

        return text

    def change_lower(text):
        """Function printing python version."""
 
        return text.lower()

    def remover(text):
        """Function printing python version."""

        text_tokens = text.split(" ")
        final_list = [word for word in text_tokens if not word in stopwords_list]
        text = ' '.join(final_list)

        return text

    def get_w2vdf(df):
        """Function printing python version."""

        df_w2v = pd.DataFrame(df[target_column]).values.tolist()
        for i in range(len(df_w2v)):
            df_w2v[i] = df_w2v[i][0].split(" ")

        return df_w2v

    stopwords_list = stopwords.words("english")

    df[[target_column]] = df[[target_column]].astype(str)
    df[target_column] = df[target_column].apply(change_lower)
    df[target_column] = df[target_column].apply(clean_data)
    df[target_column] = df[target_column].apply(remover)
    
    w2v_df = get_w2vdf(df)

    return w2v_df
