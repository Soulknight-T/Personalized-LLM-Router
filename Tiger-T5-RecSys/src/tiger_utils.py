import json
import logging
import torch
import random
from collections import defaultdict
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# --- 全局常量 ---
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
NUM_SPECIAL_TOKENS = 3

# --- 数据集类 (含采样逻辑) ---
class TigerDatasetV2(Dataset):
    """
    加载 .codes.jsonl 文件。
    如果指定 max_samples_per_user，则对每个 user_idx 进行分组并随机采样。
    """
    
    def __init__(self, filepath, max_samples_per_user: Optional[int] = None):
        self.data = []
        raw_data = []
        
        logging.info(f"正在加载数据集: {filepath}...")
        if max_samples_per_user:
            logging.info(f"--> 采样模式开启: 每个 User 最多保留 {max_samples_per_user} 条 (随机)")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        raw_data.append(json.loads(line))
            
            # --- 核心采样逻辑 ---
            if max_samples_per_user is not None:
                # 1. 按 User 分组
                user_groups = defaultdict(list)
                for item in raw_data:
                    u_idx = item['user_idx']
                    user_groups[u_idx].append(item)
                
                logging.info(f"原始数据共 {len(raw_data)} 条，包含 {len(user_groups)} 个 User。正在采样...")
                
                sampled_data = []
                for u_idx, items in user_groups.items():
                    if len(items) > max_samples_per_user:
                        # 随机抽取 N 条
                        sampled_data.extend(random.sample(items, max_samples_per_user))
                    else:
                        # 不足 N 条则保留全部
                        sampled_data.extend(items)
                
                self.data = sampled_data
                logging.info(f"采样完成。数据量从 {len(raw_data)} 减少到 {len(self.data)}。")
            else:
                # 不采样
                self.data = raw_data
                logging.info(f"加载完成: 共 {len(self.data)} 条 (无采样)。")

        except FileNotFoundError:
            logging.error(f"错误: 找不到文件 {filepath}")
        except Exception as e:
            logging.error(f"加载出错: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- 词表偏移计算 ---
def get_offsets(num_quantizers: int, codebook_size: int, user_map: Dict) -> Dict[str, int]:
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
        
    return {
        "user": user_offset,
        "query": query_offsets,
        "llm": llm_offsets,
        "total": current_offset,
        "codebook_size": codebook_size 
    }

# --- 训练用的 Collate 函数 ---
def collate_fn_v2(batch: List[Dict], offsets: Dict, num_quantizers: int):
    user_offset = offsets['user']
    query_offsets = offsets['query']
    llm_offsets = offsets['llm']
    
    encoder_input_ids_list = []
    decoder_input_ids_list = []
    decoder_labels_list = []
    
    llm_codes_map = offsets.get('llm_codes_map', {})
    llm_offsets_tensor = torch.tensor(llm_offsets, dtype=torch.long)
    
    for item in batch:
        # 1. Encoder: [BOS, User, Query..., EOS]
        user_token = torch.tensor([user_offset + item['user_idx']])
        query_codes_flat = torch.tensor(item['query_codes'], dtype=torch.long)
        
        num_turns = len(query_codes_flat) // num_quantizers
        if num_turns > 0:
            query_codes_2d = query_codes_flat.view(num_turns, num_quantizers)
            query_offsets_tensor = torch.tensor(query_offsets, dtype=torch.long)
            query_codes_seq = (query_codes_2d + query_offsets_tensor[None, :]).flatten()
        else:
            query_codes_seq = torch.empty(0, dtype=torch.long)
            
        encoder_input = torch.cat([
            torch.tensor([BOS_TOKEN]), 
            user_token, 
            query_codes_seq, 
            torch.tensor([EOS_TOKEN])
        ])
        encoder_input_ids_list.append(encoder_input)
        
        # 2. Decoder: [BOS, LLM...] / Label: [LLM..., EOS]
        # 仅当数据包含 llm_idx 且在 map 中时生成 (预测时可能不需要)
        if 'llm_idx' in item:
            llm_idx_str = str(item['llm_idx'])
            if llm_idx_str in llm_codes_map:
                llm_codes = torch.tensor(llm_codes_map[llm_idx_str], dtype=torch.long)
                llm_codes_offset = llm_codes + llm_offsets_tensor
                
                decoder_input = torch.cat([torch.tensor([BOS_TOKEN]), llm_codes_offset])
                decoder_label = torch.cat([llm_codes_offset, torch.tensor([EOS_TOKEN])])
                
                decoder_input_ids_list.append(decoder_input)
                decoder_labels_list.append(decoder_label)

    encoder_input_ids = pad_sequence(encoder_input_ids_list, batch_first=True, padding_value=PAD_TOKEN)
    encoder_attention_mask = (encoder_input_ids != PAD_TOKEN).long()
    
    if decoder_input_ids_list:
        decoder_input_ids = pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=PAD_TOKEN)
        decoder_labels = pad_sequence(decoder_labels_list, batch_first=True, padding_value=PAD_TOKEN)
    else:
        decoder_input_ids, decoder_labels = None, None
        
    return encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_labels

# --- 解码预测 ---
def decode_prediction_v2(predicted_sequence, llm_codes_map, offsets, num_quantizers):
    llm_offsets = offsets['llm']
    codebook_size = offsets['codebook_size']
    batch_predicted_codes = []
    
    for seq in predicted_sequence:
        predicted_codes_for_seq = []
        for token_id in seq.cpu().numpy():
            if token_id in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
                continue
            for q_idx in range(num_quantizers):
                offset = llm_offsets[q_idx]
                if offset <= token_id < (offset + codebook_size):
                    predicted_codes_for_seq.append(token_id - offset)
                    break
        batch_predicted_codes.append(predicted_codes_for_seq[:num_quantizers])
    return batch_predicted_codes