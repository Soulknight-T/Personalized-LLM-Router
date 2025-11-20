import os
import sys
import logging
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score

# --- 路径设置 ---
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modules.tiger_model import TigerT5
from src.train_tiger import get_offsets, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_ROOT = os.path.join(project_root, 'data')
MODELS_ROOT = os.path.join(project_root, 'saved_models')

# --- 1. 专用 Dataset: 直接读取 Winner 和 Loser ---
class PairwiseDataset(Dataset):
    def __init__(self, filepath):
        self.data = []
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到文件: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                self.data.append(json.loads(line))
        
        logging.info(f"Loaded {len(self.data)} pairs from {filepath}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- 2. 专用 Collate Function ---
def collate_fn_pairwise(batch, offsets, num_quantizers):
    user_offset = offsets.get('user')
    query_offsets = offsets.get('query')
    llm_offsets = offsets.get('llm')
    
    enc_input_list = []
    winner_label_list = []
    loser_label_list = []
    
    for item in batch:
        # --- A. Encoder Input ---
        u_token = torch.tensor([user_offset + item['user_idx']], dtype=torch.long)
        
        q_codes_flat = torch.tensor(item['query_codes'], dtype=torch.long)
        num_turns = len(q_codes_flat) // num_quantizers
        
        if num_turns > 0:
            q_codes_2d = q_codes_flat.view(num_turns, num_quantizers)
            q_offsets_tensor = torch.tensor(query_offsets, dtype=torch.long)
            q_codes_seq = (q_codes_2d + q_offsets_tensor[None, :]).flatten()
        else:
            q_codes_seq = torch.tensor([], dtype=torch.long)
            
        enc_input = torch.cat([
            torch.tensor([BOS_TOKEN]), 
            u_token, 
            q_codes_seq, 
            torch.tensor([EOS_TOKEN])
        ])
        enc_input_list.append(enc_input)
        
        # --- B. Decoder Targets ---
        llm_off_t = torch.tensor(llm_offsets, dtype=torch.long)
        
        win_c = torch.tensor(item['winner_codes'], dtype=torch.long) + llm_off_t
        lose_c = torch.tensor(item['loser_codes'], dtype=torch.long) + llm_off_t
        
        winner_label_list.append(torch.cat([win_c, torch.tensor([EOS_TOKEN])]))
        loser_label_list.append(torch.cat([lose_c, torch.tensor([EOS_TOKEN])]))

    enc_ids = pad_sequence(enc_input_list, batch_first=True, padding_value=PAD_TOKEN)
    enc_mask = (enc_ids != PAD_TOKEN).long()
    
    win_labels = torch.stack(winner_label_list)
    lose_labels = torch.stack(loser_label_list)
    
    return enc_ids, enc_mask, win_labels, lose_labels

# --- 3. 计算得分函数 ---
def calculate_sequence_score(model, enc_ids, enc_mask, target_labels, device):
    B, L = target_labels.shape
    bos_col = torch.full((B, 1), BOS_TOKEN, device=device, dtype=torch.long)
    dec_input = torch.cat([bos_col, target_labels[:, :-1]], dim=1)
    
    outputs = model(
        encoder_input_ids=enc_ids,
        encoder_attention_mask=enc_mask,
        decoder_input_ids=dec_input
    )
    logits = outputs.logits
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=PAD_TOKEN)
    logits_flat = logits.reshape(-1, logits.size(-1))
    labels_flat = target_labels.reshape(-1)
    
    loss = loss_fct(logits_flat, labels_flat)
    loss = loss.view(B, L).sum(dim=1)
    
    return -loss 

def main():
    parser = argparse.ArgumentParser(description="Explicit Pairwise Evaluation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="best_model.pth")
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_encoder_layers", type=int, default=6)
    parser.add_argument("--n_decoder_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed', args.dataset)
    TEST_PAIRS_PATH = os.path.join(PROCESSED_DIR, 'test_pairs.codes.jsonl')
    
    ckpt_path = os.path.join(MODELS_ROOT, args.dataset, "tiger_t5_v2", args.ckpt_name)
    if not os.path.exists(ckpt_path):
         ckpt_path = os.path.join(MODELS_ROOT, args.ckpt_name)

    logging.info(f"Model: {ckpt_path}")
    logging.info(f"Data: {TEST_PAIRS_PATH}")

    try:
        with open(os.path.join(PROCESSED_DIR, 'user_map.json'), 'r') as f: user_map = json.load(f)
        with open(os.path.join(PROCESSED_DIR, 'llm_map.json'), 'r') as f: llm_map = json.load(f)
    except FileNotFoundError:
        logging.error("Missing map files.")
        return

    offsets = get_offsets(args.num_quantizers, args.codebook_size, user_map)
    
    test_ds = PairwiseDataset(TEST_PAIRS_PATH)
    if len(test_ds) == 0:
        logging.error("测试集为空")
        return

    collate = partial(collate_fn_pairwise, offsets=offsets, num_quantizers=args.num_quantizers)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate)

    model = TigerT5(
        unified_vocab_size=offsets['total'],
        d_model=args.d_model,
        n_head=args.n_head,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        decoder_start_token_id=BOS_TOKEN,
        pad_token_id=PAD_TOKEN,
        eos_token_id=EOS_TOKEN
    ).to(device)
    
    logging.info("Loading weights...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    correct_pairs = 0
    total_pairs = 0
    
    # 用于 AUC 计算
    y_true = []   # [1, 0, 1, 0 ...]
    y_scores = [] # [score_win, score_lose, score_win, score_lose ...]
    
    logging.info("Starting Pairwise Evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            enc_ids, enc_mask, win_labels, lose_labels = [x.to(device) for x in batch]
            
            # 计算分数
            score_win = calculate_sequence_score(model, enc_ids, enc_mask, win_labels, device)
            score_lose = calculate_sequence_score(model, enc_ids, enc_mask, lose_labels, device)
            
            # Accuracy: 比较大小
            batch_correct = (score_win > score_lose).sum().item()
            correct_pairs += batch_correct
            total_pairs += enc_ids.size(0)
            
            # AUC Data Collection: Winner=1, Loser=0
            # 需要从 GPU 转到 CPU numpy
            s_win_np = score_win.cpu().numpy()
            s_lose_np = score_lose.cpu().numpy()
            
            # 为每一对添加两个样本
            for sw, sl in zip(s_win_np, s_lose_np):
                y_scores.extend([sw, sl])
                y_true.extend([1, 0])

    # 计算指标
    acc = correct_pairs / total_pairs if total_pairs > 0 else 0.0
    auc = roc_auc_score(y_true, y_scores) if len(y_true) > 0 else 0.0
    
    # 单行日志输出
    logging.info(f"[Result] Dataset: {args.dataset} | Pairs: {total_pairs} | Acc: {acc:.4f} | AUC: {auc:.4f}")

if __name__ == "__main__":
    main()