from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from utils import *
from sentence_spans import canonical_span_config, quran_index_directory

@st.cache_resource
def get_model(index_path, model_name, selected_device):
    return VectorSearchDeployment(index_path, model_name, selected_device)

# Set the title and a description
st.title("Find Dalil From Quran and Hadith")

sources = [ "Hadith", "Quran"]

# Create a dropdown widget
selected_option = st.selectbox("Select a source:", sources)
selected_model = "nomic-ai/nomic-embed-text-v2-moe"
selected_method = 'best_match'
selected_device = 'cpu'
selected_doctype = 'preprocessed' if selected_option=='hadith' else 'original'

model_path = f"{selected_model.split('/')[-1]}_{selected_doctype}_{selected_device}"
span_mode, span_sentences = canonical_span_config(os.environ.get('SPAN_MODE', 'whole'), os.environ.get('SPAN_SENTENCES', '1'))
quran_directory = quran_index_directory(span_mode, span_sentences)
index_dicts = {"Quran": f'vector_databases/{model_path}/{quran_directory}', 'Hadith': f'vector_databases/{model_path}/hadith', 'Both': f'vector_databases/{model_path}/all'}

index_path = index_dicts[selected_option]

input_query = st.text_area("", "Write your query")
if selected_model == "nomic-ai/nomic-embed-text-v2-moe":
    input_query = f"search_query: {input_query}"
    print(f"input query: {input_query}")
    
num_of_results = st.slider("Number of results", min_value=1, max_value=500, value=5, step=5)

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
            clean_text = deployment.display_document(result).page_content.replace('`', '') if selected_option == 'Quran' else result.metadata['original_text'].replace('`', '')
            clean_text = clean_text.replace('search_document: ', '')
            st.write(f'**{clean_text}**')
            st.markdown(f'Relevance Score: **{results[i][1]*100:.2f}**%') #normalize_l2(results[i][1]) if using IndexFlatL2 instead

            source = ''
            for key, value in result.metadata.items():
                source += f"{' '.join(key.split('_'))}: {value}, "
            
            source = source[:-2]
            
            st.markdown(f'<p style="color: gray;">{source}</p>', unsafe_allow_html=True)
            