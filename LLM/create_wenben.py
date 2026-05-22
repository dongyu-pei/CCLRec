import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import torch

def generate_aligned_npy(csv_path, output_dir, model_path):

    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
    except Exception as e:
        return

    df.columns = [c.lower().strip() for c in df.columns]
    possible_id_cols = ['MovieID']
    id_col = None
    for col in possible_id_cols:
        if col in df.columns:
            id_col = col
            break

    if id_col is None:
        id_col = df.columns[0]


    df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
    

    initial_len = len(df)
    df.dropna(subset=[id_col], inplace=True)

    item_ids = df[id_col].values.astype(int)

    text_cols = ['title', 'genres', 'year', 'genre']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        s_model = SentenceTransformer(model_path, device=device)
    except Exception as e:

        return

    sentences = []

    for _, row in df.iterrows():

        t = row['title'] if 'title' in df.columns else 'Unknown'

        y = row['year'] if 'year' in df.columns else ''
        if isinstance(y, float) or isinstance(y, int):
            y = str(int(y))

        if 'genres' in df.columns:
            g = row['genres']
        elif 'genre' in df.columns:
            g = row['genre']
        else:
            g = ''

        text = f"Title: {t}. Year: {y}. Genres: {g}."
        sentences.append(text)

    embeddings = s_model.encode(
        sentences, 
        batch_size=128, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )


    max_id = item_ids.max()
    dim = embeddings.shape[1]
    

    

    feat_matrix = np.zeros((max_id + 1, dim), dtype=np.float32)
    

    feat_matrix[item_ids] = embeddings


    output_path = os.path.join(output_dir, 'item_feat.npy')
    np.save(output_path, feat_matrix)

if __name__ == "__main__":

    DATA_DIR = ' '
    CSV_FILE = os.path.join(DATA_DIR, 'item_attribute.csv')
    LOCAL_EMBEDDING_PATH = " "
    generate_aligned_npy(CSV_FILE, DATA_DIR, LOCAL_EMBEDDING_PATH)