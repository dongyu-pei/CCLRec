import os
import time
import pickle
import pandas as pd
import requests
import json
import numpy as np
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


FILE_PATH = ""
VLLM_LOCAL_URL = ""
LOCAL_EMBEDDING_PATH = ""

PROFILING_DICT_PATH = os.path.join(FILE_PATH, "augmented_user_profiling_dict")
EMBEDDING_OUTPUT_PATH = os.path.join(FILE_PATH, "augmented_user_init_embedding")


BATCH_SIZE = 64  
LLM_TEMPERATURE = 0.5
MAX_RETRIES = 3

LOCAL_EMBEDDING_PATH_ABS = os.path.abspath(LOCAL_EMBEDDING_PATH)



s_model = SentenceTransformer(
    LOCAL_EMBEDDING_PATH_ABS,
    local_files_only=True,
    use_auth_token=False,
    cache_folder=None
)



def construct_prompting(item_attribute, item_list):
    history_string = "User's movie viewing history (ID | Year | Title | Genres):\n"
    process_list = item_list[-50:] if len(item_list) > 50 else item_list
    for index in process_list:
        try:
            row = item_attribute.loc[index]
            year = row['Year']
            title = row['Title']
            genre = row['Genres']
            history_string += f"- ID: {index} | Year: {year} | Title: {title} | Genres: {genre}\n"
        except KeyError:
            continue

    prompt_intro = (
        "You are a professional user portrait analyst. "
        "Based on the user's viewing history below, generate a structured user profile. "
        "Infer preferences based on genres, directors, and release years.\n"
        "Rules:\n"
        "1. No 'unknown' fields; make educated guesses.\n"
        "2. Do not output Chinese.\n"
        "3. Output JSON only.\n"
    )

    output_format = (
        "\nResponse Requirement: Output a valid JSON object strictly following this format:\n"
        "{\n"
        "  'summary': 'Short summary of viewing habits',\n"
        "  'age': 'Estimated age range (e.g., 20-30)',\n"
        "  'gender': 'Estimated gender',\n"
        "  'liked_genres': 'Top preferred genres',\n"
        "  'disliked_genres': 'Inferred disliked genres',\n"
        "  'preferred_era': 'Preferred release years',\n"
        "  'explanation': 'Brief reasoning for the profile'\n"
        "}\n"
    )

    prompt = prompt_intro + history_string + output_format
    return prompt

def LLM_request(prompt):
    headers = {'Content-Type': 'application/json'}
    final_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    data = {
        "prompt": final_prompt,
        "max_tokens": 512,
        "temperature": LLM_TEMPERATURE,
        "stop": ["<|im_end|>", "<|endoftext|>"]
    }
    
    for i in range(MAX_RETRIES):
        try:
            
            response = requests.post(VLLM_LOCAL_URL, headers=headers, json=data, timeout=180)
            response.raise_for_status()
            result = response.json()
            text_output = result['text'][0]
            if text_output.startswith(final_prompt):
                text_output = text_output[len(final_prompt):]
            return text_output.strip()
        except Exception:
            time.sleep(1)
            if i == MAX_RETRIES - 1:
                return None
            
def process_single_user(index, item_attribute, item_list):

    if not item_list:
        return (index, "This user has no historical behavior data.", "empty_history_fallback")
        
    try:
        prompt = construct_prompting(item_attribute, item_list)
        response = LLM_request(prompt)
        
        if response and "{" in response and "}" in response:
            return (index, response, "success")
        else:

            fallback_text = f"User profile generation failed based on history items: {item_list[:5]}..."
            final_text = response if (response and len(str(response)) > 5) else fallback_text
            
            return (index, final_text, "format_error_fallback")
            
    except Exception as e:
        return (index, "Error generating profile.", f"error: {str(e)}")

# ========================== main ==========================

if __name__ == "__main__":

    
    item_attr_path = os.path.join(FILE_PATH, 'item_attribute.csv')
    toy_item_attribute = pd.read_csv(item_attr_path) 

    mat_file_path = os.path.join(FILE_PATH, 'train_mat')
    
    if os.path.exists(mat_file_path):

        try:
            with open(mat_file_path, 'rb') as f:
                train_mat = pickle.load(f)
        except:

            train_mat = sp.load_npz(mat_file_path)
    else:
        mat_file_path_npz = mat_file_path + ".npz"
        if os.path.exists(mat_file_path_npz):

            train_mat = sp.load_npz(mat_file_path_npz)
        else:
            exit()

    train_mat = train_mat.tocsr()
    adjacency_list_dict = {}
    for u in range(train_mat.shape[0]):
        items = train_mat.indices[train_mat.indptr[u]:train_mat.indptr[u+1]]
        adjacency_list_dict[u] = items.tolist()
        
    total_users = len(adjacency_list_dict)

    augmented_user_profiling_dict = {}
    if os.path.exists(PROFILING_DICT_PATH):
        with open(PROFILING_DICT_PATH, 'rb') as f:
            augmented_user_profiling_dict = pickle.load(f)

    all_users = list(adjacency_list_dict.keys())
    pending_users = [u for u in all_users if u not in augmented_user_profiling_dict or augmented_user_profiling_dict[u] is None]

    if pending_users:

        success_cnt = 0
        save_step = 20 
        
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = {
                executor.submit(process_single_user, uid, toy_item_attribute, adjacency_list_dict[uid]): uid 
                for uid in pending_users
            }
            
            pbar = tqdm(as_completed(futures), total=len(pending_users), unit="user")
            count = 0
            for future in pbar:
                uid, response, status = future.result()
                count += 1
                
                if status == "success" or status == "format_error":
                    augmented_user_profiling_dict[uid] = response
                    success_cnt += 1
                else:
                    augmented_user_profiling_dict[uid] = None
                
                pbar.set_description(f"Success: {success_cnt}")
                
                if count % save_step == 0:
                    with open(PROFILING_DICT_PATH, 'wb') as f:
                        pickle.dump(augmented_user_profiling_dict, f)
        
        with open(PROFILING_DICT_PATH, 'wb') as f:
            pickle.dump(augmented_user_profiling_dict, f)

    valid_users = []
    valid_texts = []
    
    for uid, text in augmented_user_profiling_dict.items():
        if text and isinstance(text, str):
            if len(text) < 5:
                text = "Average movie fan."
            valid_users.append(uid)
            valid_texts.append(text)
        else:

            valid_users.append(uid)
            valid_texts.append("Unknown user profile.")
            
    if valid_texts:

        embeddings = s_model.encode(valid_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        
        user_embedding_dict = {}
        for i, uid in enumerate(valid_users):
            user_embedding_dict[uid] = embeddings[i]
            
        with open(EMBEDDING_OUTPUT_PATH, 'wb') as f:
            pickle.dump(user_embedding_dict, f)
