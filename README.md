# i23-2643-NLP-Assignment2

Natural Language Processing - Assignment 2  
BBC Urdu Neural NLP Pipeline | PyTorch from scratch

---

## Requirements

```bash
pip install torch numpy scikit-learn matplotlib seaborn nbformat
```

## Input Files

Place these three files in the same folder as the notebook:

| File | Purpose |
|------|---------|
| `cleaned.txt` | Primary training corpus (all parts) |
| `raw.txt` | Ablation baseline (Part 1, condition C2) |
| `metadata.json` | Article metadata and topic labels (Part 3) |

---

## How to Reproduce

### Part 1 - Word Embeddings

Run cells in order from **Data Loading** through **Four-Condition Comparison**.

- Builds TF-IDF matrix -> saved as `embeddings/tfidf_matrix.npy`
- Builds PPMI matrix -> saved as `embeddings/ppmi_matrix.npy`
- Trains Skip-gram Word2Vec (5 epochs, batch 512, d=100)
- Saves averaged embeddings -> `embeddings/embeddings_w2v.npy`
- Saves vocabulary -> `embeddings/word2idx.json`

### Part 2 - Sequence Labeling

Run cells from **Dataset Preparation** through **Ablation Study**.

- Annotates 500 sentences with POS (rule-based) and NER (BIO + gazetteer)
- Trains 2-layer BiLSTM (frozen + fine-tuned embeddings) for POS
- Trains BiLSTM + CRF with Viterbi decoding for NER
- Saves models -> `models/bilstm_pos.pt`, `models/bilstm_ner.pt`
- Saves CoNLL files -> `data/pos_train.conll`, `data/pos_test.conll`, etc.

### Part 3 - Transformer Classifier

Run cells from **Part 3 Dataset Preparation** through **BiLSTM vs Transformer Comparison**.

- Assigns 5 topic labels from `metadata.json`
- Trains custom Transformer encoder (4 heads, 4 layers, d=128)
- Saves model -> `models/transformer_cls.pt`

---

## Output Structure

```
embeddings/
  tfidf_matrix.npy
  ppmi_matrix.npy
  embeddings_w2v.npy
  word2idx.json
models/
  bilstm_pos.pt
  bilstm_ner.pt
  transformer_cls.pt
data/
  pos_train.conll
  pos_test.conll
  ner_train.conll
  ner_test.conll
```

---

## Notes

- All models use `device = cuda if available else cpu` - enable GPU on Colab via Runtime -> Change runtime type -> T4 GPU for ~5× speedup.
- `embed_dim = embeddings_w2v.shape[1]` must be used instead of the variable `d` in all BiLSTM constructors (d gets overwritten during C4 training).
- Total runtime: ~30 min on GPU, ~90-120 min on CPU.
