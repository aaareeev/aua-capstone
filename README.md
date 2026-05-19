# ArevNet — Adaptive Recurrent Embedding with Variable-scale Network

**Capstone Project · American University of Armenia (AUA) · May 2026**

**Student:** Arevik Melikyan | **Supervisor:** Varazdat Stepanyan | **Collaborator:** Aleksandr Hayrapetyan

Multi-scale temporal recommendation model for CTR prediction in C2C marketplaces, evaluated on MerRec, Amazon Electronics, and Amazon Books.

---

## Datasets

Datasets are large and not included in this repo. Download each one manually and update `DATA_ROOT` in Cell 1 of each notebook to match your local path.

| Dataset | Link | Size |
|---|---|---|
| MerRec | https://huggingface.co/datasets/mercari-us/merrec | ~500 GB |
| Amazon Electronics (reviews) | http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz | ~472 MB compressed |
| Amazon Electronics (metadata) | http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz | ~200 MB compressed |
| Amazon Books | http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz | ~3 GB compressed |

**MerRec:**
```bash
pip install huggingface_hub
huggingface-cli download mercari-us/merrec --repo-type dataset --local-dir <your_path>/merrec
```

**Amazon Electronics & Books:**
```bash
# Electronics
curl -C - -O --output-dir <your_path>/amazon_electronics \
  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
curl -C - -O --output-dir <your_path>/amazon_electronics \
  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d <your_path>/amazon_electronics/*.gz

# Books
curl -C - -O --output-dir <your_path>/amazon_books \
  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz
gzip -d <your_path>/amazon_books/*.gz
```

---

## Setup

```bash
conda create -n capstone python=3.11 -y
conda activate capstone
pip install -r requirements.txt
```

---

## Results

| Dataset | AUC | vs. DIEN |
|---|---|---|
| MerRec | 0.8529 | — |
| Amazon Electronics | 0.7792 | ✅ Matches |
| Amazon Books | 0.8798 | ✅ Surpasses (0.8453) |
