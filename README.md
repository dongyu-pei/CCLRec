## Dependencies

pip install -r requirements.txt

## Usage

LLM in this folder are used to generate item embeddings and user embeddings for recommendation experiments.

Specifically, the pipeline consists of the following steps:
1. `create_image.py` is used to process raw item images and generate image-level features.
2. `creat_item_embedding.py` loads visual and textual item information and constructs item embeddings based on the processed features.
3. `creat_user_embedding.py` learns user embeddings from user–item interaction data using the generated item embeddings.

Running the corresponding scripts will execute the full embedding generation pipeline.
The generated item embeddings and user embeddings can be directly used for downstream model training and evaluation.

## Download the Netflix Dataset

🚀🚀 We provide the processed data (i.e., CF training data & basic user-item interactions, original multi-modal data including images and text of items, encoded visual/textual features, and LLM-augmented text/embeddings). 🌹 We hope to contribute to our community and facilitate your research 🚀🚀  

**Note:** Our dataset follows the preprocessing and construction pipeline from [LLMRec](https://dl.acm.org/doi/pdf/10.1145/3616855.3635853), i.e., it is based on the multimodal Netflix dataset released by them.  

- **Netflix:** [Google Drive Netflix 🌟 (Image & Text)](https://drive.google.com/drive/folders/1qrDUsJFtVfIlMSKeReZ_kAo4u97I5XOW)

  

## Download the ML-1M Dataset

🚀🚀 We provide the processed data for ML-1M, including CF training/test splits, basic user-item interactions, original movie posters, encoded visual/textual features, and LLM-augmented text/embeddings. 🌹  

**Note:** Our ML-1M dataset is based on the training and test splits from the paper [CoLaKG](https://dl.acm.org/doi/pdf/10.1145/3726302.3729932).  All visual features and textual embeddings are added by us to ensure consistency with our multimodal setup.  

- **ML-1M:** [Google Drive ML-1M 🌟 (Image & Text)](https://drive.google.com/drive/folders/1qrDUsJFtVfIlMSKeReZ_kAo4u97I5XOW)

## Dataset Overview

We evaluate our model on two widely used datasets: **Netflix** and **MovieLens-1M (ML-1M)**.

- **Netflix**: We directly use the preprocessed and released multimodal Netflix dataset from [Wei et al., 2024](https://arxiv.org/abs/xxxx.xxxxx), which augments items in the Netflix Prize data with crawled movie posters and rich metadata such as titles, release years, and genres.  

- **MovieLens-1M (ML-1M)**: For this benchmark,  we adopt the same multimodal data construction pipeline to acquire the corresponding movie posters, ensuring consistency with the Netflix dataset.

### Dataset Statistics

| Statistic              | Netflix | ML-1M   |
| ---------------------- | ------- | ------- |
| #Users                 | 13,187  | 6,040   |
| #Items                 | 17,366  | 3,260   |
| #Interactions          | 70,778  | 998,539 |
| Density                | 0.0308% | 5.06%   |
| Avg. Interactions/User | 5.37    | 165.33  |
| Avg. Interactions/Item | 4.08    | 306.31  |

## Running the Code

**You can run the code：**

**Using Python command directly:**

`python main.py --dataset netflix --batch_size 64 --lr 0.001`

You can replace `--dataset` with `Ml-1M` to run on the ML-1M dataset, and adjust other hyperparameters as needed.

## Acknowledgement

The structure of this code is largely based on [MMSSL](https://github.com/HKUDS/MMSSL), [LATTICE](https://github.com/CRIPAC-DIG/LATTICE), [MICRO](https://github.com/CRIPAC-DIG/MICRO),[LLMRec](https://github.com/HKUDS/LLMRec). Thank them for their work.

