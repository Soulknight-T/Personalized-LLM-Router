from __future__ import annotations

import os
import argparse
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any

import yaml
import random
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import trange

from graph_han import HAN_GNN, PreferencePredictor 
from util_han import (
    build_graph_from_jsonl,
    load_pairs_from_jsonl,
    InteractionUnit,
    NodeDict,
    _default_device,
)

def _preconfigure_env_for_determinism(seed: Optional[int]) -> None:
    if seed is None: return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

def set_global_seed(seed: int, cuda_deterministic: bool = True, num_threads: Optional[int] = None) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

@torch.no_grad()
def evaluate_ranking(
    pairs: List[Tuple[InteractionUnit, InteractionUnit]], 
    gnn: HAN_GNN,
    predictor: PreferencePredictor,
    nodes_x_dict: NodeDict,
    metapath_edges_dev: Dict[Tuple[str, str, str], torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    gnn.eval()
    predictor.eval()
    
    n_pairs = len(pairs)
    if n_pairs == 0:
        return {"acc": 0.0, "auc": 0.5, "loss": 0.0}

    node_embeddings = gnn(nodes_x_dict, metapath_edges_dev)
    user_emb = node_embeddings["user"]
    query_emb = nodes_x_dict["query"] 
    llm_emb = nodes_x_dict["llm"]

    all_p1 = []
    all_p2 = []

    for i in range(0, n_pairs, batch_size):
        batch_pairs = pairs[i : i + batch_size]
        if not batch_pairs: continue

        batch_u1  = user_emb[ [p[0][0] for p in batch_pairs] ]
        batch_q1  = query_emb[ [p[0][2] for p in batch_pairs] ]
        batch_l1  = llm_emb[ [p[0][3] for p in batch_pairs] ]
        
        batch_u2  = user_emb[ [p[1][0] for p in batch_pairs] ]
        batch_q2  = query_emb[ [p[1][2] for p in batch_pairs] ]
        batch_l2  = llm_emb[ [p[1][3] for p in batch_pairs] ]
        
        p1 = predictor(batch_u1, batch_q1, batch_l1)
        p2 = predictor(batch_u2, batch_q2, batch_l2)
        
        all_p1.append(p1)
        all_p2.append(p2)
    
    all_p1 = torch.cat(all_p1)
    all_p2 = torch.cat(all_p2)

    acc = (all_p1 > all_p2).float().mean().item()
    
    preds = torch.cat([all_p1, all_p2])
    labels = torch.cat([torch.ones_like(all_p1), torch.zeros_like(all_p2)])
    
    auc = 0.5
    if len(torch.unique(labels)) > 1:
        try:
            auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
        except ValueError: pass
    
    loss = -F.logsigmoid(all_p1 - all_p2).mean().item()

    return {"acc": acc, "auc": auc, "loss": loss}

def train_model(
    train_jsonl_path: str,
    valid_jsonl_path: str,
    test_jsonl_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    val_epoch: int,
    ckpt_path: str,
    emb_dim: int, 
    gnn_hidden_dim: int,
    gnn_heads: int,
    pred_hidden_dim: int,
    pred_heads: int,
    dropout: float,
    dataset_name: str,
) -> dict:
    
    _preconfigure_env_for_determinism(seed)
    set_global_seed(seed)
    device = _default_device()
    (
        config,
        nodes,
        metapath_edges,
        mapping_dicts,
        device,
        train_pairs
    ) = build_graph_from_jsonl(train_jsonl_path)

    valid_pairs = load_pairs_from_jsonl(valid_jsonl_path, mapping_dicts)

    emb_dim = config.emb_dim
    print(f"  - emb_dim: {emb_dim}")

    nodes_x_dict = {k: v.to(device) for k, v in nodes.items()}
    metapath_edges_dev = {k: v.to(device) for k, v in metapath_edges.items()}

    METAPATHS_TO_USE = [("llm", "meta_LQU", "user"), ("llm", "meta_LSU", "user")]
    METAPATHS_TO_USE = [m for m in METAPATHS_TO_USE if m in metapath_edges]
    
    GNN_OUT_DIM = emb_dim 
    gnn_hidden_dim_calc = emb_dim // gnn_heads
    
    gnn = HAN_GNN(
        metapath_list=METAPATHS_TO_USE,
        in_dim=emb_dim, 
        hidden_dim=gnn_hidden_dim_calc,
        out_dim=GNN_OUT_DIM,
        heads=gnn_heads,
        dropout=dropout
    ).to(device)
    
    predictor = PreferencePredictor(
        user_dim=GNN_OUT_DIM,
        query_dim=emb_dim,
        llm_dim=emb_dim,
        hidden_dim=pred_hidden_dim,
        num_heads=pred_heads,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        list(gnn.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    n_train = len(train_pairs)
    
    print("--- Training Start ---")
    best_val_auc = -1.0
    best_ckpt_path = os.path.join(ckpt_path, f"best_model_han_{dataset_name}.pt")

    for epoch in range(epochs):
        random.shuffle(train_pairs)
        
        gnn.train()
        predictor.train()
        
        total_loss = 0.0
        all_p1_train = []
        all_p2_train = []
        
        batch_iterator = range(0, n_train, batch_size)
        
        for i in batch_iterator:
            batch_pairs = train_pairs[i : i + batch_size]
            if not batch_pairs: continue

            node_embeddings = gnn(nodes_x_dict, metapath_edges_dev)
            user_emb = node_embeddings["user"]
            query_emb = nodes_x_dict["query"]
            llm_emb = nodes_x_dict["llm"]

            batch_u1 = user_emb[[p[0][0] for p in batch_pairs]]
            batch_q1 = query_emb[[p[0][2] for p in batch_pairs]]
            batch_l1 = llm_emb[[p[0][3] for p in batch_pairs]]

            batch_u2 = user_emb[[p[1][0] for p in batch_pairs]]
            batch_q2 = query_emb[[p[1][2] for p in batch_pairs]]
            batch_l2 = llm_emb[[p[1][3] for p in batch_pairs]]

            p1 = predictor(batch_u1, batch_q1, batch_l1)
            p2 = predictor(batch_u2, batch_q2, batch_l2)

            loss = -F.logsigmoid(p1 - p2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_p1_train.append(p1.detach())
            all_p2_train.append(p2.detach())

        if not all_p1_train: continue
        
        scheduler.step()

        avg_loss = total_loss / (len(batch_iterator) + 1e-6)
        all_p1_train = torch.cat(all_p1_train)
        all_p2_train = torch.cat(all_p2_train)
        t_acc = (all_p1_train > all_p2_train).float().mean().item()

        val_metrics = evaluate_ranking(valid_pairs, gnn, predictor, nodes_x_dict, metapath_edges_dev, batch_size, device)
        current_val_auc = val_metrics['auc']

        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            print(f"  [Ep {epoch+1}] >>> New Best AUC: {best_val_auc:.4f} | Saving...")
            torch.save({
                "gnn": gnn.state_dict(),
                "pred": predictor.state_dict(),
                "mapping": mapping_dicts
            }, best_ckpt_path)

        if (epoch + 1) % 10 == 0:
             print(f"Ep {epoch+1}: Loss={avg_loss:.4f} | Train Acc={t_acc:.4f} | Val AUC={current_val_auc:.4f}")
             

    print("\n--- Testing ---")
    if os.path.exists(best_ckpt_path):
        print(f"Loading best model from {best_ckpt_path}...")
        ckpt = torch.load(best_ckpt_path)
        gnn.load_state_dict(ckpt["gnn"])
        predictor.load_state_dict(ckpt["pred"])
    
    if os.path.exists(test_jsonl_path):
        (
            t_config, 
            t_nodes, 
            t_metapath_edges, 
            t_mappings,
            _, 
            _
        ) = build_graph_from_jsonl(test_jsonl_path)
        
        real_test_pairs = load_pairs_from_jsonl(test_jsonl_path, t_mappings)
        
        if len(real_test_pairs) > 0:
            t_nodes_dev = {k: v.to(device) for k, v in t_nodes.items()}
            t_metapath_edges_dev = {k: v.to(device) for k, v in t_metapath_edges.items()}
            
            valid_test_metapaths = {k: v for k, v in t_metapath_edges_dev.items() if k in METAPATHS_TO_USE}
            test_metrics = evaluate_ranking(
                real_test_pairs, 
                gnn, 
                predictor, 
                t_nodes_dev, 
                valid_test_metapaths, 
                batch_size, 
                device
            )
            print(f"TEST RESULTS: Acc={test_metrics['acc']:.4f} | AUC={test_metrics['auc']:.4f}")
            return {"auc": test_metrics['auc']}
        else:
            print("[Inductive Test] No valid preference pairs found in test file.")
            return {"auc": 0.5}
    else:
        print(f"Test file not found: {test_jsonl_path}")
        return {"auc": 0.5}

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="./data")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints_han")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_epoch", type=int, default=10)
    
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--gnn_heads", type=int, default=4)
    parser.add_argument("--pred_hidden_dim", type=int, default=256)
    parser.add_argument("--pred_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    args = parser.parse_args()
    if args.config_path:
        with open(args.config_path, 'r') as f:
            args.__dict__.update(yaml.safe_load(f))
            
    os.makedirs(args.ckpt_path, exist_ok=True)
    
    train_model(
        train_jsonl_path=os.path.join(args.dataset_root, args.dataset_name, "train.jsonl"),
        valid_jsonl_path=os.path.join(args.dataset_root, args.dataset_name, "valid.jsonl"),
        test_jsonl_path=os.path.join(args.dataset_root, args.dataset_name, "test.jsonl"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_epoch=args.val_epoch,
        ckpt_path=args.ckpt_path,
        emb_dim=args.emb_dim,
        gnn_hidden_dim=args.emb_dim // args.gnn_heads,
        gnn_heads=args.gnn_heads,
        pred_hidden_dim=args.pred_hidden_dim,
        pred_heads=args.pred_heads,
        dropout=args.dropout,
        dataset_name=args.dataset_name,
    )

if __name__ == "__main__":
    main()