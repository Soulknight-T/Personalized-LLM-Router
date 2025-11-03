import os
import json
import sys
import logging
import argparse
import random
import pandas as pd
from tqdm import tqdm
from typing import Dict, Set, List
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
random.seed(42)

def parse_raw_data(filepath: str) -> List[Dict]:
    results = []
    skipped = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Parsing {os.path.basename(filepath)}"):
            try:
                data = json.loads(line)
                qid, model, judge = data['question_id'], str(data['model']), str(data['judge'])
                history = []
                for turn, entry in enumerate(data['conversation']):
                    history.append(entry['query'])
                    results.append({
                        'question_id': qid,
                        'turn': turn,
                        'llm_id_str': model,
                        'judge': judge,
                        'rating': entry.get('rating', 0.0),
                        'query_history': history.copy()
                    })
            except Exception:
                skipped += 1
    logging.info(f"Parsed {filepath}: {len(results)} turns, skipped {skipped}")
    return results

def create_map(data: List[Dict], key: str) -> Dict[str, int]:
    items = sorted(set(str(d[key]) for d in data))
    if not items:
        logging.error(f"Empty data for key '{key}'")
        sys.exit(1)
    mapping = {x: i for i, x in enumerate(items)}
    logging.info(f"Created map for {key}: {len(mapping)} entries")
    return mapping

def find_unique_winners(data: List[Dict]) -> List[Dict]:
    df = pd.DataFrame(data)
    winners = df.loc[df.groupby(['question_id', 'turn', 'judge'])['rating'].idxmax()]
    unique = {}
    for _, row in winners.iterrows():
        key = (json.dumps(row['query_history']), row['judge'])
        if key not in unique:
            unique[key] = {
                'query_history': row['query_history'],
                'llm_id_str': row['llm_id_str'],
                'judge': row['judge']
            }
    logging.info(f"Unique winners: {len(unique)}")
    return list(unique.values())

def write_generative_file(data: List[Dict], llm_map: Dict[str, int], user_map: Dict[str, int], out_path: str) -> Set[str]:
    queries = set()
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f"Writing {os.path.basename(out_path)}"):
            llm_idx = llm_map.get(item['llm_id_str'])
            user_idx = user_map.get(item['judge'])
            if llm_idx is None or user_idx is None:
                continue
            payload = {
                'query_history': item['query_history'],
                'llm_idx': llm_idx,
                'user_idx': user_idx
            }
            f.write(json.dumps(payload) + '\n')
            queries.add(json.dumps(item['query_history']))
    return queries

def write_corpus_files(llm_map: Dict[str, int], user_map: Dict[str, int], queries: Set[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'llm_map.json'), 'w', encoding='utf-8') as f:
        json.dump({i: n for n, i in llm_map.items()}, f, indent=2)
    with open(os.path.join(out_dir, 'user_map.json'), 'w', encoding='utf-8') as f:
        json.dump({i: n for n, i in user_map.items()}, f, indent=2)
    with open(os.path.join(out_dir, 'llm_names_corpus.txt'), 'w', encoding='utf-8') as f:
        for i in sorted(llm_map.values()):
            name = [k for k, v in llm_map.items() if v == i][0]
            f.write(name + '\n')
    with open(os.path.join(out_dir, 'queries_corpus.txt'), 'w', encoding='utf-8') as f:
        for q in tqdm(queries, desc="Writing queries corpus"):
            for t in json.loads(q):
                f.write(t + '\n')
    logging.info("Corpus files written")

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for TIGER (keep original split)")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    dataset = args.dataset
    logging.info(f"Processing dataset: {dataset}")

    raw_dir = os.path.join(DATA_ROOT, 'raw', dataset)
    out_dir = os.path.join(DATA_ROOT, 'processed', dataset)
    os.makedirs(out_dir, exist_ok=True)

    files = {k: os.path.join(raw_dir, f"{k}.jsonl") for k in ['train', 'valid', 'test']}
    all_data = []
    split_data = {}

    for split, path in files.items():
        if not os.path.exists(path):
            logging.warning(f"Missing file: {path}")
            continue
        data = parse_raw_data(path)
        split_data[split] = data
        all_data.extend(data)

    if not all_data:
        logging.error("No data found")
        sys.exit(1)

    llm_map = create_map(all_data, 'llm_id_str')
    user_map = create_map(all_data, 'judge')

    out_files = {k: os.path.join(out_dir, f"{k}.jsonl") for k in split_data}
    q_train = set()
    for split, data in split_data.items():
        winners = find_unique_winners(data)
        queries = write_generative_file(winners, llm_map, user_map, out_files[split])
        if split == 'train':
            q_train = queries

    write_corpus_files(llm_map, user_map, q_train, out_dir)
    logging.info("Done")

if __name__ == "__main__":
    main()
