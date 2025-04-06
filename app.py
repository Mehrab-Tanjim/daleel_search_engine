from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
import streamlit as st
import math
from numpy import dot
from numpy.linalg import norm

class VectorSearchDeployment:
    def __init__(self, index_path, model_name, device):
        #Load the data from faiss
        st = time.time()
        
        model_kwargs = {'device': device, 'trust_remote_code':True}
        encode_kwargs = {'normalize_embeddings': True, "max_seq_length": 8192}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        self.db = FAISS.load_local(index_path, self.embeddings, normalize_L2=True, allow_dangerous_deserialization=True)
        et = time.time() - st
        print(f'Loading database took {et} seconds.')

    def search(self, method, query, k=10): 
        if method == 'best_match':
            results = self.db.similarity_search_with_score_by_vector(
                self.embeddings.embed_query(query),
                k=k
            )
        elif method == 'best_match_dedup':
            results = self.relevance_search_dedup(query, k)
        elif method == 'mmr':
            results = self.db.max_marginal_relevance_search_with_score_by_vector(self.embeddings.embed_query(query), k=k, fetch_k=k*2)
        
        return results

    def relevance_search_dedup(self, query, k=10, similarity_threshold=0.95):
        """
        Search for most relevant results with deduplication
        Args:
            query (str): Search query
            k (int): Number of results to return
            similarity_threshold (float): Cosine similarity threshold for deduplication (0-1)
        Returns:
            list: Deduplicated search results with scores
        """
        # Get initial results using similarity search (more than k to allow for dedup)
        initial_k = k * 3
        results = self.db.similarity_search_with_score_by_vector(
            self.embeddings.embed_query(query),
            k=initial_k
        )
        
        # Deduplication process
        deduped_results = []
        seen_embeddings = []
        
        for doc, score in results:
            doc_embedding = self.embeddings.embed_query(doc.page_content)
            is_duplicate = False
            
            # Check similarity with already selected documents
            for seen_emb in seen_embeddings:
                # Calculate cosine similarity
                cos_sim = dot(doc_embedding, seen_emb) / (norm(doc_embedding) * norm(seen_emb))
                if cos_sim > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduped_results.append((doc, score))
                seen_embeddings.append(doc_embedding)
            
            # Stop when we have enough unique results
            if len(deduped_results) >= k:
                break
        
        return deduped_results[:k]

def normalize_l2(score):
    return 1 - score/math.sqrt(2)

@st.cache_resource
def get_model(index_path, model_name, selected_device):
    return VectorSearchDeployment(index_path, model_name, selected_device)

# Set the title and a description
st.title("Find Dalil From Quran and Hadith")

options = [ "Hadith", "Quran", "Both"]
model_names = ["Alibaba-NLP/gte-multilingual-base",  "sentence-transformers/LaBSE", "sentence-transformers/all-mpnet-base-v2", "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-small", 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'] #, "sentence-transformers/all-mpnet-base-v2" ]
search_methods = ['best_match', 'best_match_dedup', 'mmr']

# Create a dropdown widget
selected_option = st.selectbox("Select a source:", options)
selected_model = st.selectbox("Select a model:", model_names)
selected_method = st.selectbox("Select a search method:", search_methods)
selected_device = st.selectbox("Select a device:", ['cuda', 'cpu'])

model_path = f"{selected_model.split('/')[-1]}_{selected_device}"
index_dicts = {"Quran": f'vector_databases/{model_path}/quran', 'Hadith': f'vector_databases/{model_path}/hadith', 'Both': f'vector_databases/{model_path}/all'}

index_path = index_dicts[selected_option]

input_query = st.text_area("", "Write your query")

num_of_results = st.slider("Number of results", min_value=1, max_value=500, value=25, step=1)

search_button = st.button("Search")


if search_button:

    deployment = get_model(index_path, selected_model, selected_device)

    # Display the user's input
    results = deployment.search(selected_method, input_query, num_of_results)

    st.write("Search Results: ")

    for i in range(len(results)):
        result = results[i][0]

        # Create two columns
        col1, col2 = st.columns([1, 19])

        # In the first column, add text and a button
        with col1:
            st.markdown(f'<p style="color: red;">{i+1}</p>', unsafe_allow_html=True)
        
        with col2:
        # Display the text
            clean_text = result.page_content.replace('`', '')
            st.write(f'**{clean_text}**')
            st.markdown(f'Relevance Score: **{normalize_l2(results[i][1])*100:.2f}**%')

            source = ''
            for key, value in result.metadata.items():
                source += f"{' '.join(key.split('_'))}: {value}, "
            
            source = source[:-2]
            
            st.markdown(f'<p style="color: gray;">{source}</p>', unsafe_allow_html=True)
            