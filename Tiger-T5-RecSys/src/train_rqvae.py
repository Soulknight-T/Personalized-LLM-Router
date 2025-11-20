import os
import sys

current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import argparse
import numpy as np
import random # Added
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from sentence_transformers import SentenceTransformer
from src.modules.rqvae_model import RQVAE
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
MODELS_ROOT = os.path.join(PROJECT_ROOT, 'saved_models')

SBERT_MODEL_NAME = 'sentence-t5-base'

# --- 新增: 设置随机种子 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Train RQ-VAE model")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--corpus_name", type=str, default="queries_corpus.txt")
    parser.add_argument("--model_name", type=str, default="query_rqvae")
    parser.add_argument("--skip_kmeans", action="store_true")
    parser.add_argument("--num_quantizers", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    # --- 新增: Seed 参数 ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # --- 应用 Seed ---
    set_seed(args.seed)
    logging.info(f"Random seed set to: {args.seed}")

    DATASET_NAME = args.dataset
    logging.info(f"Training RQ-VAE '{args.model_name}' on dataset '{DATASET_NAME}'")

    PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed', DATASET_NAME)
    MODEL_SAVE_DIR = os.path.join(MODELS_ROOT, DATASET_NAME, args.model_name)
    CORPUS_PATH = os.path.join(PROCESSED_DIR, args.corpus_name)
    EMBEDDING_CACHE_PATH = os.path.join(PROCESSED_DIR, f"{os.path.splitext(args.corpus_name)[0]}.emb.{SBERT_MODEL_NAME}.npy")
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if os.path.exists(EMBEDDING_CACHE_PATH):
        all_embeddings = np.load(EMBEDDING_CACHE_PATH)
    else:
        if not os.path.exists(CORPUS_PATH):
            logging.error(f"Corpus not found: {CORPUS_PATH}")
            sys.exit(1)
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        logging.info(f"Encoding {len(lines)} sentences with SBERT")
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
        all_embeddings = sbert_model.encode(lines, show_progress_bar=True, device=device)
        np.save(EMBEDDING_CACHE_PATH, all_embeddings)

    logging.info(f"Embeddings shape: {all_embeddings.shape}")

    dataset = TensorDataset(torch.tensor(all_embeddings, dtype=torch.float32))
    batch_size = min(args.batch_size, len(dataset))

    if len(dataset) < batch_size * 2 or len(dataset) <= 16:
        train_dataset = dataset
        val_dataset = None
    else:
        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        # 注意: set_seed 必须在 random_split 之前调用以保证划分一致
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    pin_memory = True if device.type == "cuda" else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    ) if val_dataset else None

    if val_loader:
        logging.info(f"Validation set size: {len(val_dataset)}")

    model = RQVAE(
        input_dim=all_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        codebook_dim=args.codebook_dim,
        num_quantizers=args.num_quantizers,
        codebook_size=args.codebook_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if not args.skip_kmeans:
        effective_codebook_size = min(args.codebook_size, len(train_dataset))
        model.quantizer.codebook_size = effective_codebook_size

        try:
            (first_batch,) = next(iter(train_loader))
            first_batch = first_batch.to(device)
            with torch.no_grad():
                model.eval()
                latent_z = model.encode(first_batch)
                model.quantizer.initialize_codebooks(latent_z)
                model.train()
            logging.info(f"K-means initialized with codebook_size={effective_codebook_size}")
        except StopIteration:
            logging.error("Empty training data, cannot init codebook")
            sys.exit(1)
        except Exception as e:
            logging.error(f"K-means initialization failed: {e}")
            sys.exit(1)
    else:
        logging.warning("Skipping K-means initialization")

    best_val_loss = float('inf')
    logging.info("Start training")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_recon = train_loss_commit = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for (batch,) in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_recon, recon_loss, commit_loss = model(batch)
            loss = recon_loss + commit_loss
            loss.backward()
            optimizer.step()
            train_loss_recon += recon_loss.item()
            train_loss_commit += commit_loss.item()
            pbar.set_postfix(recon=f"{recon_loss.item():.4f}", commit=f"{commit_loss.item():.4f}")

        avg_train_recon = train_loss_recon / len(train_loader)
        avg_train_commit = train_loss_commit / len(train_loader)
        avg_train_total = avg_train_recon + avg_train_commit

        if val_loader:
            model.eval()
            val_loss_recon = val_loss_commit = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    x_recon, recon_loss, commit_loss = model(batch)
                    val_loss_recon += recon_loss.item()
                    val_loss_commit += commit_loss.item()
            avg_val_total = (val_loss_recon + val_loss_commit) / len(val_loader)
            logging.info(f"[Epoch {epoch:02d}] Train: {avg_train_total:.6f} | Val: {avg_val_total:.6f}")
            current_loss = avg_val_total
        else:
            logging.info(f"[Epoch {epoch:02d}] Train total: {avg_train_total:.6f}")
            current_loss = avg_train_total

        if current_loss < best_val_loss:
            best_val_loss = current_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"Saved best model: {MODEL_SAVE_PATH} ({best_val_loss:.6f})")

    logging.info(f"Training finished: '{args.model_name}'")

if __name__ == "__main__":
    main()