# Vector Space Model Information Retrieval System

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [About VSM](#about-vector-space-models)

## Installation
```bash
git clone https://github.com/ashboy27/Vector_Space_Model.git
cd Vector_Space_Model
pip install -r Code/requirements.txt
python -m nltk.downloader punkt wordnet
```

## Usage
# After placing files:
cd Code
streamlit run app.py
## About Vector Space Models

The Vector Space Model (VSM) represents text documents and queries as vectors in a high-dimensional space where:
- Each dimension corresponds to a unique term from the document collection
- Terms are weighted using TF-IDF (Term Frequency-Inverse Document Frequency)
- Document relevance is measured using cosine similarity between vectors

**Key Mathematical Formulations**:
