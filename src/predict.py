import os
import sys
import logging
import argparse
import json
import torch
from tqdm import tqdm
from typing import Dict, Tuple

current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modules.tiger_model import TigerT5
from src.train_tiger import TigerDatasetV2, get_offsets
from src.train_tiger import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_ROOT = os.path.join(project_root, 'data')
MODELS_ROOT = os.path.join(project_root, 'saved_models')

@torch.no_grad()
def prepare_input_v2(item: Dict, offsets: Dict, num_quantizers: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    user_token = [offsets['USER_TOKEN_OFFSET'] + item['user_idx']]
    query_tokens = [q + offsets['query_offsets'][i % num_quantizers] for i, q in enumerate(item['query_codes'])]
    input_ids = torch.tensor([user_token + query_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

@torch.no_grad()
def decode_prediction_v2(predicted_sequence: torch.Tensor, llm_codes_map: Dict, offsets: Dict, num_quantizers: int) -> int:
    if predicted_sequence.dim() > 1:
        predicted_sequence = predicted_sequence.squeeze(0)
    if predicted_sequence[0] == BOS_TOKEN:
        predicted_sequence = predicted_sequence[1:]
    pred_end_mask = (predicted_sequence == EOS_TOKEN) | (predicted_sequence == PAD_TOKEN)
    pred_end_idx = torch.where(pred_end_mask, 1, 0).argmax() or predicted_sequence.size(0)
    predicted_tokens = predicted_sequence[:pred_end_idx]

    if len(predicted_tokens) != num_quantizers:
        predicted_tokens = predicted_tokens[:num_quantizers] if len(predicted_tokens) > num_quantizers else \
                           torch.tensor(predicted_tokens.cpu().tolist() + [-1]*(num_quantizers - len(predicted_tokens)))

    pred_raw_codes = []
    for i in range(num_quantizers):
        token = predicted_tokens[i].item()
        raw_code = token - offsets['llm_offsets'][i]
        if not (0 <= raw_code < offsets.get('codebook_size', 256)):
            raw_code = -1
        pred_raw_codes.append(raw_code)

    best_match_idx = -1
    min_distance = float('inf')
    for llm_idx_str, true_codes in llm_codes_map.items():
        distance = sum(p != t for p, t in zip(pred_raw_codes, true_codes))
        if distance < min_distance:
            min_distance = distance
            best_match_idx = int(llm_idx_str)
        if min_distance == 0:
            break
    return best_match_idx

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="TIGER T5 v2 prediction")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--eval_split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--output_file", type=str, default="test_predictions.jsonl")
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_encoder_layers", type=int, default=2)
    parser.add_argument("--n_decoder_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    args = parser.parse_args()

    PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed', args.dataset)
    MODEL_SAVE_DIR = os.path.join(MODELS_ROOT, args.dataset, 'tiger_t5_v2')
    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
    EVAL_CODES_PATH = os.path.join(PROCESSED_DIR, f"{args.eval_split}.codes.jsonl")
    OUTPUT_PATH = os.path.join(MODEL_SAVE_DIR, args.output_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    with open(os.path.join(PROCESSED_DIR, 'user_map.json')) as f:
        user_map = json.load(f)
    with open(os.path.join(PROCESSED_DIR, 'llm_map.json')) as f:
        llm_map = json.load(f)
    with open(os.path.join(PROCESSED_DIR, 'llm_codes.json')) as f:
        llm_codes_map = json.load(f)

    offsets = get_offsets(len(user_map), args.num_quantizers, args.codebook_size)
    offsets['codebook_size'] = args.codebook_size
    unified_vocab_size = offsets['unified_vocab_size']

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
    )
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found: {MODEL_PATH}")
        sys.exit(1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    eval_dataset = TigerDatasetV2(EVAL_CODES_PATH)
    logging.info(f"Start prediction: {len(eval_dataset)} samples")

    results = []
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        for item in tqdm(eval_dataset, desc=f"Predict {args.eval_split}"):
            input_tensor, att_mask = prepare_input_v2(item, offsets, args.num_quantizers, device)
            pred_seq = model.generate(input_ids=input_tensor, attention_mask=att_mask,
                                      max_length=args.num_quantizers + 2, num_beams=1)
            predicted_llm_idx = decode_prediction_v2(pred_seq.cpu(), llm_codes_map, offsets, args.num_quantizers)

            true_llm_idx = item['llm_idx']
            user_idx = item['user_idx']
            output_data = {
                'query_codes': item['query_codes'],
                'user_name': user_map.get(str(user_idx), f"User_{user_idx}"),
                'true_llm_name': llm_map.get(str(true_llm_idx), f"LLM_{true_llm_idx}"),
                'predicted_llm_name': llm_map.get(str(predicted_llm_idx), "FAILED"),
                'is_correct': (predicted_llm_idx == true_llm_idx)
            }
            f_out.write(json.dumps(output_data) + '\n')
            results.append(output_data)

    total_correct = sum(r['is_correct'] for r in results)
    total_count = len(results)
    val_accuracy = total_correct / total_count if total_count > 0 else 0.0

    logging.info(f"Prediction finished. Total: {total_count}, Correct: {total_correct}, Accuracy: {val_accuracy*100:.2f}%")
    logging.info(f"Predictions saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
