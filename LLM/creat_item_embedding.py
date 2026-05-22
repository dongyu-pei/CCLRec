import os
import time
import pickle
import pandas as pd
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


FILE_PATH = ""
VLLM_LOCAL_URL = ""
LOCAL_EMBEDDING_PATH = ""

AUGMENTED_CSV_PATH = os.path.join(FILE_PATH, "augmented_item_attribute_movielens.csv")
ATTRIBUTE_EMBEDDING_PATH = os.path.join(FILE_PATH, "augmented_attribute_embedding_dict")
BATCH_SIZE = 80
LLM_TEMPERATURE = 0.3
MAX_RETRIES = 3


def construct_augmentation_prompt(title, year, genre):

    prompt = (
        f"Movie Info: Title: \"{title}\", Year: {year}.\n"
        "Task: Identify the Director, Country of Origin, and Primary Language of this movie.\n"
        "Output Format strict requirement: Please output ONLY the information in this specific format:\n"
        "Director::Country::Language\n"
        "Example: Steven Spielberg::USA::English\n"
        "Do not output any other text, introduction, or context. If unknown, make an educated guess."
    )
    final_prompt = f"<|im_start|>system\nYou are a movie knowledge database.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return final_prompt

def LLM_request_info(prompt):

    headers = {'Content-Type': 'application/json'}
    data = {
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": LLM_TEMPERATURE,
        "stop": ["<|im_end|>", "\n\n"]
    }
    
    for i in range(MAX_RETRIES):
        try:
            response = requests.post(VLLM_LOCAL_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            text_output = result['text'][0]
            

            if text_output.startswith(prompt):
                text_output = text_output[len(prompt):]
            
            return text_output.strip()
        except Exception:
            time.sleep(1)
    return "Unknown::Unknown::Unknown"

def process_single_item_augmentation(row):

    item_id = row['MovieID']
    title = row['Title']
    year = row['Year']
    genre = row['Genres']
    
    prompt = construct_augmentation_prompt(title, year, genre)
    response = LLM_request_info(prompt)

    try:
        parts = response.split('::')
        if len(parts) >= 3:
            return item_id, parts[0].strip(), parts[1].strip(), parts[2].strip()
        else:

            return item_id, response, "Unknown", "Unknown"
    except:
        return item_id, "Unknown", "Unknown", "Unknown"

def run_augmentation_pipeline(item_df):

    print(f"\n[1/3]  {len(item_df)})...")

    results = {}

    
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = []
        for index, row in item_df.iterrows():
            futures.append(executor.submit(process_single_item_augmentation, row))
            
        for future in tqdm(as_completed(futures), total=len(item_df), unit="item"):
            item_id, director, country, language = future.result()
            results[item_id] = {
                'director': director,
                'country': country,
                'language': language
            }

    director_list = []
    country_list = []
    language_list = []
    
    for index, row in item_df.iterrows():
        uid = row['MovieID']
        res = results.get(uid, {'director': 'Unknown', 'country': 'Unknown', 'language': 'Unknown'})
        director_list.append(res['director'])
        country_list.append(res['country'])
        language_list.append(res['language'])
        
    item_df['director'] = director_list
    item_df['country'] = country_list
    item_df['language'] = language_list

    item_df.to_csv(AUGMENTED_CSV_PATH, index=False)

    return item_df


def run_embedding_pipeline(aug_df, s_model):

    target_columns = ['Year', 'Title', 'director', 'country', 'language']
    final_embedding_dict = {}
    
    item_ids = aug_df['MovieID'].tolist()
    
    for col in target_columns:
        final_embedding_dict[col] = {}

        texts = aug_df[col].fillna("Unknown").astype(str).tolist()
        embeddings = s_model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)

        col_dict = {}
        for i, uid in enumerate(item_ids):
            col_dict[uid] = embeddings[i]
            
        final_embedding_dict[col] = col_dict

    with open(ATTRIBUTE_EMBEDDING_PATH, 'wb') as f:
        pickle.dump(final_embedding_dict, f)


# ========================== main ==========================

if __name__ == "__main__":

    item_attr_path = os.path.join(FILE_PATH, 'item_attribute.csv')

    df_item = pd.read_csv(item_attr_path)
    if os.path.exists(AUGMENTED_CSV_PATH):
        df_augmented = pd.read_csv(AUGMENTED_CSV_PATH)
    else:
        df_augmented = run_augmentation_pipeline(df_item)


        s_model = SentenceTransformer(LOCAL_EMBEDDING_PATH)

    run_embedding_pipeline(df_augmented, s_model)