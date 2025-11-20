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
import numpy as np
import math
import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# --- ↓↓↓ 导入路径修正 ↓↓↓ ---
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modules.tiger_model import TigerT5 

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_ROOT = os.path.join(project_root, 'data')
MODELS_ROOT = os.path.join(project_root, 'saved_models')

# --- 1. TIGER 词表和特殊 Token (不变) ---
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
NUM_SPECIAL_TOKENS = 3 # PAD, BOS, EOS

# --- 2. 【v2 修正 + 历史采样】TIGER 数据集 ---
class TigerDatasetV2(Dataset):
    """
    【v2 修正】加载 .codes.jsonl
    【新增】支持 max_history (训练时随机采样，验证时固定采样)
    """
    def __init__(self, filepath: str, max_history: int = 10, is_train: bool = True):
        self.data = []
        self.user_pool = defaultdict(list)  # 用于存储每个用户的所有 query codes
        self.max_history = max_history
        self.is_train = is_train
        
        logging.info(f"正在加载 TIGER v2 数据集: {filepath} (max_history={max_history}, is_train={is_train})...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    self.data.append(item)
                    
                    # --- 构建用户历史池 ---
                    u_idx = item.get('user_idx')
                    q_codes = item.get('query_codes')
                    if u_idx is not None and q_codes is not None:
                        self.user_pool[u_idx].append(q_codes)
                        
            logging.info(f"成功加载 {len(self.data)} 个样本。User pool size: {len(self.user_pool)}")
            
        except FileNotFoundError:
            logging.error(f"错误: 找不到 .codes 文件: {filepath}")
            logging.error("请先运行 convertID.py。")
        except Exception as e:
            logging.error(f"加载 .codes 文件时出错: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        user_idx = item['user_idx']
        current_query_codes = item['query_codes']
        
        # --- 历史记录采样逻辑 ---
        all_history = self.user_pool[user_idx]
        
        sampled_history = []
        if len(all_history) > 0:
            if self.is_train:
                # 训练模式：随机采样 max_history 条 (类似 GNN 的 DropEdge)
                k = min(len(all_history), self.max_history)
                sampled_history = random.sample(all_history, k)
            else:
                # 验证/测试模式：确定性采样 (取最近/最后 max_history 条)
                # 假设文件顺序即时间顺序
                sampled_history = all_history[-self.max_history:]
        
        # 展平历史记录
        flat_history_codes = [code for seq in sampled_history for code in seq]
        
        # 拼接：[历史记录 Codes] + [当前 Query Codes]
        final_input_codes = flat_history_codes + current_query_codes
        
        # 返回字典 (与 collate_fn_v2 兼容)
        return {
            'user_idx': user_idx,
            'llm_idx': item['llm_idx'],
            'query_codes': final_input_codes 
        }

# --- 3. 【v2 修正】TIGER 词表 (Offsets) ---
def get_offsets(num_quantizers: int, codebook_size: int, user_map: Dict) -> Dict[str, int]:
    """计算 T5 统一词表中各个部分的起始偏移量"""
    
    current_offset = NUM_SPECIAL_TOKENS 
    
    user_offset = current_offset
    current_offset += len(user_map)
    
    query_offsets = []
    for _ in range(num_quantizers):
        query_offsets.append(current_offset)
        current_offset += codebook_size
    
    llm_offsets = []
    for _ in range(num_quantizers):
        llm_offsets.append(current_offset)
        current_offset += codebook_size
        
    total_vocab_size = current_offset
    
    return {
        "user": user_offset,
        "query": query_offsets,
        "llm": llm_offsets,
        "total": total_vocab_size
    }


# --- 4. 【v2 修正】TIGER 批处理 (Collate) ---
def collate_fn_v2(batch: List[Dict], offsets: Dict, num_quantizers: int):
    """
    【v2 修正】将数据批处理为 TIGER (T5) 的输入张量
    """
    
    user_offset = offsets.get('user')
    query_offsets = offsets.get('query')
    llm_offsets = offsets.get('llm')
    
    encoder_input_ids_list = []
    
    for item in batch:
        # a. User Token
        user_idx = item['user_idx']
        user_token = torch.tensor([user_offset + user_idx]) # (1,)
        
        # b. Query Tokens (这里已经是拼接了历史记录的长序列)
        query_codes_flat = torch.tensor(item['query_codes'], dtype=torch.long) # (Total_Seq_Len,)
        
        # 计算 turn 数量 (包括历史 turn 和当前 turn)
        num_turns = len(query_codes_flat) // num_quantizers
        
        if num_turns == 0: 
            query_codes_seq = torch.empty(0, dtype=torch.long)
        else:
            # Reshape to (Turns, Quantizers)
            query_codes_2d = query_codes_flat.view(num_turns, num_quantizers)
            query_offsets_tensor = torch.tensor(query_offsets, dtype=torch.long) # (Q,)
            # Add Offsets
            query_codes_2d_offset = query_codes_2d + query_offsets_tensor[None, :] 
            # Flatten back
            query_codes_seq = query_codes_2d_offset.flatten() 
        
        # c. 组装 T5 输入: [BOS, User, History..., CurrentQuery, EOS]
        encoder_input = torch.cat([
            torch.tensor([BOS_TOKEN]),
            user_token,
            query_codes_seq,
            torch.tensor([EOS_TOKEN])
        ], dim=0)
        
        encoder_input_ids_list.append(encoder_input)

    # Decoder 部分 (不变)
    llm_offsets_tensor = torch.tensor(llm_offsets, dtype=torch.long) 
    llm_codes_map = offsets['llm_codes_map'] 
    
    decoder_input_ids_list = []
    decoder_labels_list = []
    
    for item in batch:
        llm_idx_str = str(item['llm_idx'])
        llm_codes = torch.tensor(llm_codes_map[llm_idx_str], dtype=torch.long) 
        llm_codes_offset = llm_codes + llm_offsets_tensor 
        
        decoder_input = torch.cat([torch.tensor([BOS_TOKEN]), llm_codes_offset], dim=0)
        decoder_label = torch.cat([llm_codes_offset, torch.tensor([EOS_TOKEN])], dim=0)
        
        decoder_input_ids_list.append(decoder_input)
        decoder_labels_list.append(decoder_label)

    encoder_input_ids = pad_sequence(encoder_input_ids_list, batch_first=True, padding_value=PAD_TOKEN)
    encoder_attention_mask = (encoder_input_ids != PAD_TOKEN).long()
    
    decoder_input_ids = pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=PAD_TOKEN)
    decoder_labels = pad_sequence(decoder_labels_list, batch_first=True, padding_value=PAD_TOKEN)
    
    return encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_labels

# --- 5. 【v2 修正】解码预测 (用于日志) ---
def decode_prediction_v2(predicted_sequence, llm_codes_map, offsets, num_quantizers):
    llm_offsets = offsets.get('llm')
    codebook_size = offsets['codebook_size']
    batch_predicted_codes = []
    for seq in predicted_sequence:
        predicted_codes_for_seq = []
        for token_id in seq.cpu().numpy():
            if token_id in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]: continue
            for q_idx in range(num_quantizers):
                offset = llm_offsets[q_idx]
                if offset <= token_id < (offset + codebook_size):
                    code = token_id - offset
                    predicted_codes_for_seq.append(code)
                    break
        batch_predicted_codes.append(predicted_codes_for_seq[:num_quantizers])
    return batch_predicted_codes


# --- 6. 训练主函数 ---
def main():
    parser = argparse.ArgumentParser(description="【v11】训练 TIGER T5 (支持历史采样)")
    parser.add_argument("--dataset", type=str, required=True)
    
    # VAE 架构
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    
    # TIGER (T5) 架构
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_encoder_layers", type=int, default=6)
    parser.add_argument("--n_decoder_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    
    # 新增: 历史记录参数
    parser.add_argument("--max_history", type=int, default=10, help="最大历史记录数")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DATASET_NAME = args.dataset
    logging.info(f"--- 开始训练 TIGER T5: '{DATASET_NAME}' (Max History: {args.max_history}) ---")

    # --- 1. 定义路径 ---
    PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed', DATASET_NAME)
    MODEL_DIR = os.path.join(MODELS_ROOT, DATASET_NAME, "tiger_t5_v2") 
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
    
    LLM_MAP_PATH = os.path.join(PROCESSED_DIR, 'llm_map.json')
    USER_MAP_PATH = os.path.join(PROCESSED_DIR, 'user_map.json')
    LLM_CODES_PATH = os.path.join(PROCESSED_DIR, 'llm_codes.json')
    
    TRAIN_CODES_PATH = os.path.join(PROCESSED_DIR, 'train.codes.jsonl')
    VALID_CODES_PATH = os.path.join(PROCESSED_DIR, 'valid.codes.jsonl')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. 加载 Maps 和 VAE Codes ---
    try:
        with open(LLM_MAP_PATH, 'r') as f: llm_map = json.load(f)
        with open(USER_MAP_PATH, 'r') as f: user_map = json.load(f)
        with open(LLM_CODES_PATH, 'r') as f: llm_codes_map = json.load(f) 
    except FileNotFoundError:
        logging.error("找不到 map/codes 文件，请先运行预处理。")
        return

    # --- 3. 计算词表 (Offsets) ---
    offsets = get_offsets(args.num_quantizers, args.codebook_size, user_map)
    UNIFIED_VOCAB_SIZE = offsets['total']
    offsets['llm_codes_map'] = llm_codes_map 
    offsets['codebook_size'] = args.codebook_size 
    
    logging.info(f"Unified Vocab Size: {UNIFIED_VOCAB_SIZE}")
    
    # --- 4. 初始化模型 (使用正确的参数) ---
    model = TigerT5(
        unified_vocab_size=UNIFIED_VOCAB_SIZE, # 这里修复了！
        d_model=args.d_model,
        n_head=args.n_head,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        decoder_start_token_id=BOS_TOKEN,
        pad_token_id=PAD_TOKEN,
        eos_token_id=EOS_TOKEN
    ).to(device)

    # --- 5. 设置数据加载器 (应用历史采样) ---
    from functools import partial
    collate_fn_bound = partial(collate_fn_v2, offsets=offsets, num_quantizers=args.num_quantizers)

    # 训练集：is_train=True (开启随机采样)
    train_dataset = TigerDatasetV2(TRAIN_CODES_PATH, max_history=args.max_history, is_train=True)
    # 验证集：is_train=False (固定采样)
    val_dataset = TigerDatasetV2(VALID_CODES_PATH, max_history=args.max_history, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_bound, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_bound, num_workers=args.num_workers)
    
    # --- 6. 训练循环 ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            enc_ids, enc_mask, dec_ids, dec_labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(encoder_input_ids=enc_ids, encoder_attention_mask=enc_mask, decoder_input_ids=dec_ids, decoder_labels=dec_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Valid
        model.eval()
        val_loss = 0
        total_code_correct = 0
        total_code_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} [Valid]")):
                enc_ids, enc_mask, dec_ids, dec_labels = [b.to(device) for b in batch]
                
                # Loss
                outputs = model(encoder_input_ids=enc_ids, encoder_attention_mask=enc_mask, decoder_input_ids=dec_ids, decoder_labels=dec_labels)
                val_loss += outputs.loss.item()
                
                # Acc
                generated = model.generate(encoder_input_ids=enc_ids, encoder_attention_mask=enc_mask, max_length=(args.num_quantizers + 2))
                decoded_batch = decode_prediction_v2(generated, llm_codes_map, offsets, args.num_quantizers)
                
                # Compare logic
                current_bs = enc_ids.size(0)
                for i in range(current_bs):
                    pred = torch.tensor(decoded_batch[i])
                    orig_idx = (batch_idx * args.batch_size) + i
                    true_llm_idx = str(val_dataset.data[orig_idx]['llm_idx'])
                    true_codes = torch.tensor(llm_codes_map[true_llm_idx])
                    
                    clen = min(len(true_codes), len(pred))
                    if clen > 0:
                        total_code_correct += (true_codes[:clen] == pred[:clen]).sum().item()
                    total_code_count += len(true_codes)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = total_code_correct / total_code_count if total_code_count else 0
        
        logging.info(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            logging.info(f"Saved best model to {MODEL_PATH}")

if __name__ == "__main__":
    main()