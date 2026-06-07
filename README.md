# Consumer Complaint Intelligence — NLP on CFPB Data

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/FinBERT-ProsusAI-orange?logo=huggingface&logoColor=white)
![spaCy](https://img.shields.io/badge/NER-spaCy_en__core__web__lg-09a3d5?logo=spacy&logoColor=white)
![BERTopic](https://img.shields.io/badge/Topics-BERTopic-7952b3)
![License](https://img.shields.io/badge/Data-US_Government_Public_Domain-green)

An end-to-end NLP pipeline for extracting structured intelligence from unstructured financial text, demonstrated on 2M+ real consumer complaint narratives from the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).

> Built as a portfolio project to demonstrate applied NLP skills relevant to financial services — adviser review analysis, compliance document screening, and product disclosure assessment.

---

## Pipeline overview

```
CFPB Narratives (2M+)
        │
        ▼
┌───────────────────┐     ┌──────────────────────┐
│  FinBERT          │     │  spaCy NER           │
│  Sentiment        │     │  + custom risk ruler │
│  pos/neg/neutral  │     │  ORG, PRODUCT,       │
│  per complaint    │     │  RISK_TERM, MONEY    │
└───────────────────┘     └──────────────────────┘
        │                          │
        └──────────┬───────────────┘
                   ▼
        ┌─────────────────────┐     ┌──────────────────────────┐
        │  BERTopic           │     │  Compliance Flag         │
        │  Unsupervised theme │     │  Detection               │
        │  discovery (UMAP +  │     │  Embedding cosine sim    │
        │  HDBSCAN)           │     │  against 6 risk categories│
        └─────────────────────┘     └──────────────────────────┘
```

---

## Techniques demonstrated

| Module | Technique | Library |
|--------|-----------|---------|
| Sentiment analysis | Financial domain BERT fine-tune | `ProsusAI/finbert` |
| Named entity recognition | Pre-trained NER + custom `span_ruler` | `spacy en_core_web_lg` |
| Topic modelling | Embedding-based clustering, no labels needed | `BERTopic`, `UMAP`, `HDBSCAN` |
| Compliance flagging | Semantic similarity to risk phrase templates | `sentence-transformers` |
| Embeddings | Sentence-level dense vectors | `all-MiniLM-L6-v2` |

---

## Compliance flag categories

Six risk categories modelled on CFPB enforcement priorities, detected via cosine similarity (threshold 0.55) rather than keyword matching — catching paraphrases that rule-based search misses:

- Undisclosed fees
- Robo-signing / document fraud
- Harassment by debt collector
- Credit reporting error
- Predatory / unfair lending
- Identity theft / unauthorized account

---

## Quickstart

### On Kaggle (recommended)
1. Open the notebook → **Copy & Edit**
2. Add dataset: **Data → Add data** → search `namigabbasov consumer complaint dataset`
3. Enable GPU: **Settings → Accelerator → GPU T4**
4. **Run All**

### Locally
```bash
git clone https://github.com/keithtang2025/cfpb-complaint-nlp.git
cd cfpb-complaint-nlp
pip install -r requirements.txt
python -m spacy download en_core_web_lg
jupyter notebook cfpb_nlp_pipeline.ipynb
```

> Set `FULL_LOAD = False` and `LOAD_N = 50_000` for a fast development run (~15 min on GPU).  
> Set `FULL_LOAD = True` for the complete 2M-row dataset (~4 GB RAM required).

---

## Requirements

```
transformers>=4.40
torch>=2.0
spacy>=3.7
bertopic>=0.16
sentence-transformers>=2.7
umap-learn>=0.5
hdbscan>=0.8
kagglehub>=0.2
pandas>=2.0
matplotlib>=3.8
seaborn>=0.13
wordcloud>=1.9
scikit-learn>=1.4
```

---

## Data

**Source:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)  
**Kaggle mirror:** [`namigabbasov/consumer-complaint-dataset`](https://www.kaggle.com/datasets/namigabbasov/consumer-complaint-dataset)  
**Coverage:** December 2011 – August 2024 · 2,023,066 complaints with narratives  
**License:** U.S. Government public domain work

---

## Project structure

```
cfpb-complaint-nlp/
├── cfpb_nlp_pipeline.ipynb   # Main notebook — all sections self-contained
├── README.md
└── requirements.txt
```

---

## Author

**Keith Tang**  

Former Investment Analytics & Strategy Lead, TriTree Capital  
[Kaggle](https://kaggle.com/tangtszkwong)
