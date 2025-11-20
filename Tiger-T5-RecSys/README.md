## 1. Install Requirements

```bash
pip install -r requirements.txt
```

## 2. Prepare Raw Data

Put the raw data into the `raw_data` folder.

## 3. Preprocess Dataset

```bash
python -m src.preprocess --dataset 'dataset_name'
```

## 4. Train RQVAE Model

```bash
python -m src.train_rqvae --dataset 'dataset_name'
```

## 5. Convert Query into SemanticID

```bash
python -m src.convertID --dataset 'dataset_name'
```

The `llm_name` will be encoded as `0,0,0,0` or `1,1,1,1` accordingly.

## 6. Train TIGER Model

```bash
python -m src.train_tiger --dataset 'dataset_name'
```

Usually it takes 15 epochs.

## 7. Run Test Dataset

```bash
python -m src.predict --dataset 'dataset_name'
```

**Note:** Only the `gsm8k` dataset works effectively since it has much more training data for the model.

