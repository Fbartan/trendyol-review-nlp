# Trendyol Review NLP Project

This repository contains scripts and resources for topic modeling and coherence evaluation of Turkish e-commerce reviews scraped from Trendyol. The following topic modeling methods are implemented and compared:

- BERTopic
- CTM (Contextualized Topic Modeling)
- Top2Vec
- LDA (Latent Dirichlet Allocation)

## Structure

- `modeling/`: Training and coherence scoring for each model
- `visualization/`: Word clouds, bar charts, topic maps
- `data/`: Raw and processed Excel data
- `utils/`: (reserved for any custom preprocessing tools)

## Requirements

```bash
pip install -r requirements.txt
```

## Output Samples

Example topic word clouds and NPMI coherence curve are included in `visualization/images/`.
