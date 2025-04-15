from collections import Counter
import math
import os
from preprocessing import preprocess_text


def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            doc_id = filename[:-4]
            print(os.path.join(directory, filename))
            with open(os.path.join(directory, filename), "r") as f:
                text = f.read()
            documents.append((doc_id, text))
    return documents


def build_indices(documents, stop_words):
    inverted_index = {}
    doc_terms = {}
    for doc_id, text in documents:
        tokens = preprocess_text(text, stop_words)
        term_freq = Counter(tokens)
        doc_terms[doc_id] = term_freq
        for term, tf in term_freq.items():
            if term not in inverted_index:
                inverted_index[term] = {"df": 1, "postings": {doc_id: tf}}
            else:
                if doc_id not in inverted_index[term]["postings"]:
                    inverted_index[term]["df"] += 1
                inverted_index[term]["postings"][doc_id] = tf
    return inverted_index, doc_terms


def compute_document_norms(inverted_index, doc_terms, N):
    document_norms = {}
    for doc_id, term_freq in doc_terms.items():
        norm_sq = 0.0
        for term, tf in term_freq.items():
            df = inverted_index.get(term, {"df": 0})["df"]
            if df == 0:
                continue
            idf = math.log(df) / N
            weight = tf * idf
            norm_sq += weight**2
        document_norms[doc_id] = math.sqrt(norm_sq) if norm_sq != 0 else 0.0
    return document_norms


def process_query(
    query, stop_words, inverted_index, doc_terms, document_norms, N, threshold
):
    preprocessed_query = preprocess_text(query, stop_words)
    term_freq_query = Counter(preprocessed_query)
    query_vector = {}
    for term, tf in term_freq_query.items():
        if term in inverted_index:
            df = inverted_index[term]["df"]
            idf = math.log(df) / N
            query_vector[term] = tf * idf
    query_norm = (
        math.sqrt(sum(w**2 for w in query_vector.values())) if query_vector else 0.0
    )
    if query_norm == 0:
        return []
    results = []
    for doc_id in doc_terms:
        dot_product = 0.0
        for term in query_vector:
            if term in doc_terms[doc_id]:
                tf_doc = doc_terms[doc_id][term]
                df = inverted_index[term]["df"]
                idf = math.log(df) / N
                doc_weight = tf_doc * idf
                query_weight = query_vector[term]
                dot_product += doc_weight * query_weight
        doc_norm = document_norms.get(doc_id, 0.0)
        similarity = (
            dot_product / (doc_norm * query_norm)
            if (doc_norm * query_norm) != 0
            else 0.0
        )
        if similarity >= threshold:
            results.append((doc_id, similarity))
    results.sort(key=lambda x: x[1], reverse=True)
    return results
