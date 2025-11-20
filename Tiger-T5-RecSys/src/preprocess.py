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
    """
    原有的逻辑：只找 Winner，用于训练集 (Train) 构造生成任务
    """
    df = pd.DataFrame(data)
    # 找到 rating 最高的行
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
    logging.info(f"Unique winners (for Train): {len(unique)}")
    return list(unique.values())

# --- 新增函数 Start: 寻找成对数据 ---
def find_pairwise_entries(data: List[Dict]) -> List[Dict]:
    """
    新逻辑：用于测试集 (Test)，找出 Winner 和 Loser 对。
    忽略平局 (Tie)。
    """
    df = pd.DataFrame(data)
    pairwise_results = []
    
    # 按 User + Question + Turn 分组
    grouped = df.groupby(['question_id', 'turn', 'judge'])
    
    for name, group in tqdm(grouped, desc="Finding pairs"):
        if len(group) < 2:
            continue # 数据不完整，无法比较
            
        # 按分数降序排列
        sorted_group = group.sort_values('rating', ascending=False)
        
        # 取最高分和最低分
        winner = sorted_group.iloc[0]
        loser = sorted_group.iloc[-1]
        
        # 如果分数相同，则是平局，跳过
        if winner['rating'] == loser['rating']:
            continue
            
        pairwise_results.append({
            'query_history': winner['query_history'],
            'judge': winner['judge'],
            'winner_llm_str': winner['llm_id_str'],
            'loser_llm_str': loser['llm_id_str'],
            'winner_rating': float(winner['rating']),
            'loser_rating': float(loser['rating'])
        })
        
    logging.info(f"Found {len(pairwise_results)} valid pairs (Winner > Loser).")
    return pairwise_results

def write_pairwise_file(data: List[Dict], llm_map: Dict[str, int], user_map: Dict[str, int], out_path: str):
    """
    写入成对数据文件 test_pairs.jsonl
    """
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f"Writing Pairwise {os.path.basename(out_path)}"):
            w_idx = llm_map.get(item['winner_llm_str'])
            l_idx = llm_map.get(item['loser_llm_str'])
            u_idx = user_map.get(item['judge'])
            
            # 确保所有 ID 都存在于 Map 中
            if w_idx is not None and l_idx is not None and u_idx is not None:
                payload = {
                    'query_history': item['query_history'],
                    'user_idx': u_idx,
                    'winner_llm_idx': w_idx,
                    'loser_llm_idx': l_idx
                }
                f.write(json.dumps(payload) + '\n')
# --- 新增函数 End ---

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
    # 写入 LLM 列表以便查看
    with open(os.path.join(out_dir, 'llm_names_corpus.txt'), 'w', encoding='utf-8') as f:
        # 确保按 ID 顺序写入
        sorted_llms = sorted(llm_map.items(), key=lambda x: x[1])
        for name, idx in sorted_llms:
            f.write(name + '\n')
            
    with open(os.path.join(out_dir, 'queries_corpus.txt'), 'w', encoding='utf-8') as f:
        for q in tqdm(queries, desc="Writing queries corpus"):
            for t in json.loads(q):
                f.write(t + '\n')
    logging.info("Corpus files written")

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for TIGER")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    logging.info(f"Random seed set to: {args.seed}")

    dataset = args.dataset
    logging.info(f"Processing dataset: {dataset}")

    raw_dir = os.path.join(DATA_ROOT, 'raw', dataset)
    out_dir = os.path.join(DATA_ROOT, 'processed', dataset)
    os.makedirs(out_dir, exist_ok=True)

    files = {k: os.path.join(raw_dir, f"{k}.jsonl") for k in ['train', 'valid', 'test']}
    all_data = []
    split_data = {}

    # 1. 解析所有原始数据
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

    # 2. 建立 Map
    llm_map = create_map(all_data, 'llm_id_str')
    user_map = create_map(all_data, 'judge')

    # 3. 处理各个 Split
    out_files = {k: os.path.join(out_dir, f"{k}.jsonl") for k in split_data}
    q_train = set()

    for split, data in split_data.items():
        # A. 标准流程：只保留 Winner (用于训练，或者 Test 的 Ranking 评估)
        winners = find_unique_winners(data)
        queries = write_generative_file(winners, llm_map, user_map, out_files[split])
        
        if split == 'train':
            q_train = queries

        # B. 测试集特殊处理：生成 Pairwise 文件 (包含 Winner 和 Loser)
        if split == 'test':
            logging.info("Processing Test Pairs (Winner vs Loser)...")
            pairs = find_pairwise_entries(data)
            pair_out_path = os.path.join(out_dir, "test_pairs.jsonl")
            write_pairwise_file(pairs, llm_map, user_map, pair_out_path)

    # 4. 写入 Corpus 信息
    write_corpus_files(llm_map, user_map, q_train, out_dir)
    logging.info("Done")

if __name__ == "__main__":
    main()