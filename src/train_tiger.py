import os
import sys
import logging
import argparse
import json
import math
from typing import List, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modules.tiger_model import TigerT5 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_ROOT = os.path.join(project_root, 'data')
MODELS_ROOT = os.path.join(project_root, 'saved_models')

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
NUM_SPECIAL_TOKENS = 3

class TigerDatasetV2(Dataset):
    def __init__(self, codes_jsonl_path):
        try:
            with open(codes_jsonl_path, 'r', encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f]
            logging.info(f"Loaded {len(self.data)} samples from {codes_jsonl_path}")
        except FileNotFoundError:
            logging.error(f"File not found: {codes_jsonl_path}")
            sys.exit(1)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TigerT5CollatorV2:
    def __init__(self, llm_codes_map: Dict, offsets: Dict, num_quantizers: int):
        self.llm_codes_map = llm_codes_map
        self.offsets = offsets
        self.num_quantizers = num_quantizers
        
    def __call__(self, batch: List[Dict]) -> Dict:
        encoder_inputs, decoder_inputs, decoder_labels = [], [], []

        for item in batch:
            user_token = [self.offsets['USER_TOKEN_OFFSET'] + item['user_idx']]
            query_tokens = [
                q + self.offsets['query_offsets'][i % self.num_quantizers]
                for i, q in enumerate(item['query_codes'])
            ]
            encoder_inputs.append(torch.tensor(user_token + query_tokens, dtype=torch.long))

            llm_codes = self.llm_codes_map[str(item['llm_idx'])]
            llm_tokens = [c + self.offsets['llm_offsets'][i] for i, c in enumerate(llm_codes)]

            decoder_inputs.append(torch.tensor([BOS_TOKEN] + llm_tokens, dtype=torch.long))
            decoder_labels.append(torch.tensor(llm_tokens + [EOS_TOKEN], dtype=torch.long))

        encoder_input_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_TOKEN)
        decoder_input_ids = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_TOKEN)
        decoder_labels = pad_sequence(decoder_labels, batch_first=True, padding_value=PAD_TOKEN)

        return {
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": (encoder_input_ids != PAD_TOKEN).long(),
            "decoder_input_ids": decoder_input_ids,
            "decoder_labels": decoder_labels,
        }

def get_offsets(num_users: int, num_quantizers: int, codebook_size: int) -> Dict:
    offsets = {}
    offsets['USER_TOKEN_OFFSET'] = NUM_SPECIAL_TOKENS
    offsets['QUERY_TOKEN_OFFSET'] = NUM_SPECIAL_TOKENS + num_users
    offsets['query_offsets'] = [offsets['QUERY_TOKEN_OFFSET'] + i * codebook_size for i in range(num_quantizers)]
    offsets['LLM_TOKEN_OFFSET'] = offsets['QUERY_TOKEN_OFFSET'] + num_quantizers * codebook_size
    offsets['llm_offsets'] = [offsets['LLM_TOKEN_OFFSET'] + i * codebook_size for i in range(num_quantizers)]
    offsets['unified_vocab_size'] = offsets['LLM_TOKEN_OFFSET'] + num_quantizers * codebook_size
    return offsets

def main():
    parser = argparse.ArgumentParser(description="Train TIGER T5 v2")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_encoder_layers", type=int, default=2)
    parser.add_argument("--n_decoder_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    DATASET_NAME = args.dataset
    logging.info(f"Start training TIGER T5: {DATASET_NAME}")

    PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed', DATASET_NAME)
    MODEL_SAVE_DIR = os.path.join(MODELS_ROOT, DATASET_NAME, 'tiger_t5_v2')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    USER_MAP_PATH = os.path.join(PROCESSED_DIR, 'user_map.json')
    LLM_MAP_PATH = os.path.join(PROCESSED_DIR, 'llm_map.json')
    LLM_CODES_PATH = os.path.join(PROCESSED_DIR, 'llm_codes.json')
    TRAIN_CODES_PATH = os.path.join(PROCESSED_DIR, 'train.codes.jsonl')
    VALID_CODES_PATH = os.path.join(PROCESSED_DIR, 'valid.codes.jsonl')
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    with open(USER_MAP_PATH) as f:
        user_map = json.load(f)
    with open(LLM_MAP_PATH) as f:
        llm_map = json.load(f)
    with open(LLM_CODES_PATH) as f:
        llm_codes_map = json.load(f)

    offsets = get_offsets(len(user_map), args.num_quantizers, args.codebook_size)
    unified_vocab_size = offsets['unified_vocab_size']

    logging.info(f"Vocab size: {unified_vocab_size}")

    model = TigerT5(
        unified_vocab_size=unified_vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        decoder_start_token_id=BOS_TOKEN,
        pad_token_id=PAD_TOKEN,
        eos_token_id=EOS_TOKEN
    ).to(device)
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_dataset = TigerDatasetV2(TRAIN_CODES_PATH)
    valid_dataset = TigerDatasetV2(VALID_CODES_PATH)
    collator = TigerT5CollatorV2(llm_codes_map, offsets, args.num_quantizers)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collator, pin_memory=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collator)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            optimizer.zero_grad()
            encoder_input_ids = batch['encoder_input_ids'].to(device)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_labels = batch['decoder_labels'].to(device)

            outputs = model(encoder_input_ids=encoder_input_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_labels=decoder_labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        total_correct = 0
        total_count = 0
        for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Valid]"):
            encoder_input_ids = batch['encoder_input_ids'].to(device)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_labels = batch['decoder_labels'].to(device)

            outputs = model(encoder_input_ids=encoder_input_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_labels=decoder_labels)
            val_loss += outputs.loss.item()

            generated_ids = model.generate(input_ids=encoder_input_ids,
                                           attention_mask=encoder_attention_mask,
                                           max_length=args.num_quantizers + 2,
                                           num_beams=1)
            pred_seq = generated_ids[:, 1:]
            true_seq = decoder_labels
            for i in range(pred_seq.size(0)):
                true_codes = true_seq[i]
                pred_codes = pred_seq[i]
                true_end = (true_codes == EOS_TOKEN) | (true_codes == PAD_TOKEN)
                pred_end = (pred_codes == EOS_TOKEN) | (pred_codes == PAD_TOKEN)
                true_len = torch.where(true_end, 1, 0).argmax() or true_codes.size(0)
                pred_len = torch.where(pred_end, 1, 0).argmax() or pred_codes.size(0)
                compare_len = min(true_len, pred_len)
                total_correct += (true_codes[:compare_len] == pred_codes[:compare_len]).sum().item()
                total_count += true_len
        avg_val_loss = val_loss / len(val_loader)
        code_acc = total_correct / total_count if total_count > 0 else 0.0

        logging.info(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f} | "
                     f"Valid PPL: {math.exp(avg_val_loss):.2f} |  Acc: {code_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"Saved new best model to {MODEL_SAVE_PATH}")

    logging.info("Training finished")

if __name__ == "__main__":
    main()
