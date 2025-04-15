import streamlit as st
import pandas as pd
from preprocessing import load_stop_words
from vsm import load_documents, build_indices, compute_document_norms, process_query
import math
from collections import Counter
from preprocessing import preprocess_text



@st.cache_data
def load_data():
    stop_words = load_stop_words('stopwords.txt')
    documents = load_documents('Abstract')
    inverted_index, doc_terms = build_indices(documents, stop_words)
    N = len(documents)
    document_norms = compute_document_norms(inverted_index, doc_terms, N)
    return inverted_index, doc_terms, document_norms, N, stop_words

def display_query_processing(query, stop_words, inverted_index, N):
    st.subheader("Query Processing Steps")
    
    with st.expander("1. Preprocessing Details"):
        st.write("**Original Query:**")
        st.code(query)
        
        processed_query = preprocess_text(query, stop_words)
        st.write("**After Processing (lowercase, stopwords removed, lemmatized):**")
        st.code(" ".join(processed_query))
    
    with st.expander("2. Term Weighting (TF-IDF Calculation)"):
        term_freq = Counter(processed_query)
        query_weights = []
        
        st.write("**Term Frequency (TF) in Query:**")
        tf_df = pd.DataFrame([(term, count) for term, count in term_freq.items()],
                            columns=["Term", "Raw Frequency"])
        st.dataframe(tf_df)
        
        st.write("**Final TF-IDF Weights (TF * IDF):**")
        for term in term_freq:
            df = inverted_index.get(term, {}).get('df', 0)
            idf = math.log(df)/N if df > 0 else 0
            query_weights.append((term, term_freq[term], df, idf, term_freq[term] * idf))
        
        weight_df = pd.DataFrame(query_weights,
                                columns=["Term", "TF", "DF", "IDF", "TF-IDF"])
        st.dataframe(weight_df.style.format({"IDF": "{:.4f}", "TF-IDF": "{:.4f}"}))
    
    with st.expander("3. Vector Space Representation"):
        st.write("**Query Vector Representation**")
        st.latex(r"\text{Query Vector} = [w_1, w_2, \ldots, w_n] \text{ where } w_i = \text{TF}_i \times \text{IDF}_i")
        
        vector_representation = "\n".join([f"{term}: {weight:.4f}" 
                                         for term, _, _, _, weight in query_weights])
        st.text(vector_representation)
        
        st.write("**Document Vectors**")
        st.write("Each document is represented similarly in the same vector space")
        st.latex(r"\text{Similarity}(q,d) = \frac{\vec{q} \cdot \vec{d}}{||\vec{q}|| \cdot ||\vec{d}||}")

def main():
    st.title("Vector Space Model Information Retrieval")
    st.markdown("""
    ### How It Works
    1. **Query Processing**: Tokenization, stopword removal, and lemmatization
    2. **TF-IDF Weighting**: Compute term importance using combined TF and IDF
    3. **Vector Comparison**: Calculate cosine similarity between query and documents
    """)
    
    inverted_index, doc_terms, document_norms, N, stop_words = load_data()

    query = st.text_input("Enter your query:")
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.05, 0.01)

    if query:
        display_query_processing(query, stop_words, inverted_index, N)
        
        results = process_query(query, stop_words, inverted_index, doc_terms,
                               document_norms, N, threshold)
        
        st.subheader("Search Results")
        if not results:
            st.warning("No documents found above the threshold")
        else:
            st.write(f"Found {len(results)} documents with similarity ≥ {threshold:.2f}")
            
            similarity_df = pd.DataFrame(results, columns=["Document ID", "Similarity"])
            st.dataframe(similarity_df.style.format({"Similarity": "{:.4f}"}))
            
            st.markdown("""
            ### Understanding the Results
            - **Similarity Score**: Ranges from 0 (no match) to 1 (perfect match)
            - **Threshold**: Only documents above this similarity score are shown
            - **Ranking**: Results sorted by descending similarity score
            """)

main()