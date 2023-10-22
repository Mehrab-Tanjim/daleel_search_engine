from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import time
import streamlit as st
import math


class VectorSearchDeployment:
    def __init__(self, index_path):
        #Load the data from faiss
        st = time.time()
        model_name = "sentence-transformers/all-mpnet-base-v2" 
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        self.db = FAISS.load_local(index_path, self.embeddings, normalize_L2=True)
        et = time.time() - st
        print(f'Loading database took {et} seconds.')

    def search(self, query, k=10): 
        results= self.db.max_marginal_relevance_search_with_score_by_vector(self.embeddings.embed_query(query), k=k, fetch_k=k*2)
        return results

def normalize_l2(score):
    return 1 - score/math.sqrt(2)

# Set the title and a description
st.title("Search Daleel")

options = ["Quran", "Hadith"]

# Create a dropdown widget
selected_option = st.selectbox("Select a source:", options)

if selected_option == 'Quran':
    index_path = 'clean-raw-quran-all-mpnet-base-v2-normalized-cuda'
else:
    index_path = 'clean-raw-hadith-all-mpnet-base-v2-normalized-cuda'

input_query = st.text_area("", "Write your query")

num_of_results = st.slider("Number of results", min_value=1, max_value=500, value=25, step=1)

search_button = st.button("Search")


if search_button:

    deployment = VectorSearchDeployment(index_path)

    # Display the user's input
    results = deployment.search(input_query, num_of_results)

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
            # st.markdown(f'Relevance Score: **{normalize_l2(results[i][1])*100:.2f}**%')

            source = ''
            for key, value in result.metadata.items():
                source += f"{' '.join(key.split('_'))}: {value}, "
            
            source = source[:-2]
            
            st.markdown(f'<p style="color: gray;">{source}</p>', unsafe_allow_html=True)
            