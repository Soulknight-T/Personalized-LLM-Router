import os
import sys

current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import logging
import argparse
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.modules.rqvae_model import RQVAE

current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_ROOT = os.path.join(project_root, 'data')
MODELS_ROOT = os.path.join(project_root, 'saved_models')
SBERT_MODEL_NAME = 'sentence-t5-base'

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Preprocess queries + generate LLM codes")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="query_rqvae")
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()
    
    DATASET_NAME = args.dataset
    logging.info(f"Processing dataset: {DATASET_NAME}")

    PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed', DATASET_NAME)
    QUERY_RQVAE_PATH = os.path.join(MODELS_ROOT, DATASET_NAME, args.model_name, 'best_model.pth')
    LLM_MAP_PATH = os.path.join(PROCESSED_DIR, 'llm_map.json')

    # --- 标准文件路径 ---
    TRAIN_IN_PATH = os.path.join(PROCESSED_DIR, "train.jsonl")
    VALID_IN_PATH = os.path.join(PROCESSED_DIR, "valid.jsonl")
    TEST_IN_PATH = os.path.join(PROCESSED_DIR, "test.jsonl")

    # --- 新增: Pairwise 文件路径 ---
    TEST_PAIRS_IN_PATH = os.path.join(PROCESSED_DIR, "test_pairs.jsonl")

    # --- 输出路径 ---
    LLM_CODES_OUT_PATH = os.path.join(PROCESSED_DIR, "llm_codes.json")
    TRAIN_OUT_PATH = os.path.join(PROCESSED_DIR, "train.codes.jsonl")
    VALID_OUT_PATH = os.path.join(PROCESSED_DIR, "valid.codes.jsonl")
    TEST_OUT_PATH = os.path.join(PROCESSED_DIR, "test.codes.jsonl")
    
    # --- 新增: Pairwise 输出路径 ---
    TEST_PAIRS_OUT_PATH = os.path.join(PROCESSED_DIR, "test_pairs.codes.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if not os.path.exists(LLM_MAP_PATH):
        logging.error(f"LLM map not found: {LLM_MAP_PATH}")
        return
    with open(LLM_MAP_PATH, 'r') as f:
        llm_map = json.load(f)

    logging.info("Loading SBERT model...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    SBERT_DIM = sbert_model.get_sentence_embedding_dimension()

    logging.info("Loading Query RQ-VAE model...")
    try:
        query_rqvae = RQVAE(
            input_dim=SBERT_DIM, hidden_dim=args.hidden_dim, codebook_dim=args.codebook_dim,
            num_quantizers=args.num_quantizers, codebook_size=args.codebook_size
        )
        query_rqvae.load_state_dict(torch.load(QUERY_RQVAE_PATH, map_location=device))
        query_rqvae = query_rqvae.to(device)
        query_rqvae.eval()
    except Exception as e:
        logging.error(f"Failed to load Query VAE: {e}")
        return

    logging.info("Generating LLM codes...")
    llm_codes_map = {}
    num_llms = len(llm_map)
    if num_llms > args.codebook_size:
        logging.warning(f"LLM count ({num_llms}) exceeds codebook size ({args.codebook_size})")
    
    # 这里生成你的 1111, 2222 等规则的 Codes
    for i in range(num_llms):
        llm_codes_map[str(i)] = [i % args.codebook_size] * args.num_quantizers

    with open(LLM_CODES_OUT_PATH, 'w') as f:
        json.dump(llm_codes_map, f, indent=2)
    logging.info(f"LLM codes saved to {LLM_CODES_OUT_PATH}")

    # --- 通用处理函数 ---
    def process_split(input_path: str, output_path: str, split_name: str):
        if not os.path.exists(input_path):
            logging.warning(f"File not found, skipping: {input_path}")
            return
        dataset = []
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                if line.strip():
                    dataset.append(json.loads(line))
        if not dataset:
            logging.warning(f"Empty file, skipping: {input_path}")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for item in tqdm(dataset, desc=f"{split_name}"):
                if 'query_history' not in item or not item['query_history']:
                    continue
                
                # 生成 Query Codes
                sbert_embs = torch.tensor(
                    sbert_model.encode(item['query_history'], device=device, show_progress_bar=False),
                    dtype=torch.float32
                ).to(device)
                latent_embs = query_rqvae.encode(sbert_embs)
                codes_flat = query_rqvae.quantizer.get_codes(latent_embs).flatten().cpu().numpy().tolist()
                
                output_data = {
                    "user_idx": item['user_idx'],
                    "llm_idx": item['llm_idx'],
                    "query_codes": codes_flat
                }
                f_out.write(json.dumps(output_data) + '\n')
        logging.info(f"{split_name} codes saved to {output_path}")

    # --- 新增: 处理 Pairwise 文件的函数 ---
    def process_pairwise_split(input_path: str, output_path: str):
        if not os.path.exists(input_path):
            logging.warning(f"Pairwise file not found (Normal if not testing): {input_path}")
            return
            
        dataset = []
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                if line.strip():
                    dataset.append(json.loads(line))
        
        if not dataset:
            return

        with open(output_path, 'w', encoding='utf-8') as f_out:
            for item in tqdm(dataset, desc="Processing Pairs"):
                if 'query_history' not in item or not item['query_history']:
                    continue

                # 1. 生成 Query Codes
                sbert_embs = torch.tensor(
                    sbert_model.encode(item['query_history'], device=device, show_progress_bar=False),
                    dtype=torch.float32
                ).to(device)
                latent_embs = query_rqvae.encode(sbert_embs)
                codes_flat = query_rqvae.quantizer.get_codes(latent_embs).flatten().cpu().numpy().tolist()

                # 2. 查找 Winner 和 Loser 的 Codes
                # preprocess.py 输出的是 llm_idx (int)，我们需要转成 str 来查 llm_codes_map
                w_idx = str(item['winner_llm_idx'])
                l_idx = str(item['loser_llm_idx'])
                
                w_codes = llm_codes_map.get(w_idx)
                l_codes = llm_codes_map.get(l_idx)
                
                if w_codes is None or l_codes is None:
                    logging.warning(f"Missing codes for LLM IDs {w_idx} or {l_idx}, skipping item.")
                    continue

                output_data = {
                    "user_idx": item['user_idx'],
                    "query_codes": codes_flat,
                    "winner_codes": w_codes,
                    "loser_codes": l_codes
                }
                f_out.write(json.dumps(output_data) + '\n')
        
        logging.info(f"Pairwise codes saved to {output_path}")

    # --- 执行处理 ---
    process_split(TRAIN_IN_PATH, TRAIN_OUT_PATH, "Train")
    process_split(VALID_IN_PATH, VALID_OUT_PATH, "Valid")
    process_split(TEST_IN_PATH, TEST_OUT_PATH, "Test")
    
    # 调用 Pairwise 处理
    process_pairwise_split(TEST_PAIRS_IN_PATH, TEST_PAIRS_OUT_PATH)

    logging.info(f"All processing completed. Outputs in {PROCESSED_DIR}")

if __name__ == "__main__":
    main()