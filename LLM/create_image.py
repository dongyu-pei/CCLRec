import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


DATA_DIR = ''
CSV_FILE = os.path.join(DATA_DIR, 'item_attribute.csv')
IMAGE_DIR = os.path.join(DATA_DIR, 'image')
OUTPUT_FILE = os.path.join(DATA_DIR, 'image_feat.npy')


LOCAL_MODEL_PATH = " "

BATCH_SIZE = 64


def load_local_model(model_path):

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        processor = CLIPProcessor.from_pretrained(model_path)

        full_model = CLIPModel.from_pretrained(model_path).vision_model.to(device)
        full_model.eval()
        
        return processor, full_model, device
    except Exception as e:

        exit()

def process_images_to_npy(csv_path, img_dir, output_path, model_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return

    df.columns = [c.strip() for c in df.columns]
    id_col = None
    for col in df.columns:
        if col.lower() == 'movieid':
            id_col = col
            break
    if not id_col:
        id_col = df.columns[0]

    df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
    df.dropna(subset=[id_col], inplace=True)
    
    all_ids = df[id_col].values.astype(int)
    max_id = all_ids.max()

    processor, model, device = load_local_model(model_path)

    try:
        embed_dim = model.config.hidden_size
    except:
        embed_dim = 768

    feat_matrix = np.zeros((max_id + 1, embed_dim), dtype=np.float32)

    batch_ids = []
    batch_images = []
    missing_ids = []

    def run_batch(_ids, _imgs):
        if not _ids:
            return
        try:
            inputs = processor(images=_imgs, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                image_embeds = outputs.pooler_output

                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                feat_matrix[_ids] = image_embeds.cpu().numpy()
        except Exception as e:
            print("error")

    for uid in tqdm(all_ids):
        img_path = os.path.join(img_dir, f"{uid}.jpg")
        
        image_obj = None
        if os.path.exists(img_path):
            try:
                image_obj = Image.open(img_path).convert('RGB')
            except:
                image_obj = None
        

        if image_obj is None:
            image_obj = Image.new('RGB', (224, 224), color='black')
            missing_ids.append(uid)
        
        batch_ids.append(uid)
        batch_images.append(image_obj)
        
        if len(batch_ids) >= BATCH_SIZE:
            run_batch(batch_ids, batch_images)
            batch_ids = []
            batch_images = []


    if batch_ids:
        run_batch(batch_ids, batch_images)

    np.save(output_path, feat_matrix)
    


if __name__ == "__main__":
    process_images_to_npy(CSV_FILE, IMAGE_DIR, OUTPUT_FILE, LOCAL_MODEL_PATH)