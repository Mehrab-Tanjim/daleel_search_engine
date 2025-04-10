from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from utils import *

@st.cache_resource
def get_model(index_path, model_name, selected_device):
    return VectorSearchDeployment(index_path, model_name, selected_device)

# Set the title and a description
st.title("Find Dalil From Quran and Hadith")

options = [ "Hadith", "Quran", "Both"]
model_names = ["nomic-ai/nomic-embed-text-v1", "nomic-ai/nomic-embed-text-v2-moe",
        "fine_tuned_models/islamqa_fine_tuned_all-mpnet-base-v2",
        "Alibaba-NLP/gte-multilingual-base",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/LaBSE",
        "intfloat/multilingual-e5-base",
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2']
search_methods = ['best_match', 'best_match_dedup', 'mmr']

# Create a dropdown widget
selected_option = st.selectbox("Select a source:", options)
selected_model = st.selectbox("Select a model:", model_names)
selected_method = st.selectbox("Select a search method:", search_methods)
selected_device = st.selectbox("Select a device:", ['cpu', 'cuda'])
selected_doctype = st.selectbox("Select a device:", ['preprocessed', 'original'])

model_path = f"{selected_model.split('/')[-1]}_{selected_doctype}_{selected_device}"
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
            st.markdown(f'Relevance Score: **{results[i][1]*100:.2f}**%') #normalize_l2(results[i][1]) if using IndexFlatL2 instead

            source = ''
            for key, value in result.metadata.items():
                source += f"{' '.join(key.split('_'))}: {value}, "
            
            source = source[:-2]
            
            st.markdown(f'<p style="color: gray;">{source}</p>', unsafe_allow_html=True)
            