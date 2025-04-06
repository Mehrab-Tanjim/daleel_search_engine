from langchain_community.docstore.document import Document
import pickle
import pandas as pd
from tqdm import tqdm

def dataframe_to_documents(df, page_content_column, meta_data_columns):
    documents = []
    print(f"Converting to documents using '{page_content_column}' as content")
    
    for _, data_point in tqdm(df.iterrows(), total=len(df)):
        if not isinstance(data_point[page_content_column], float):
            metadata = {
                key: str(data_point[key]).strip() for key in meta_data_columns
            }
            documents.append(Document(
                page_content=data_point[page_content_column].strip(),
                metadata=metadata
            ))
    return documents

def save_documents(documents, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(documents, f)
    print(f"Saved {len(documents)} documents to {output_path}")

# Quran
quran_dataset = pd.read_excel(r'Quran\holy_quran_dataset.xlsx')
quran_docs = dataframe_to_documents(
    quran_dataset,
    page_content_column='EnglishTranslation',
    meta_data_columns=['SurahNameEnglish', 'SurahNo', 'AyahNo', 'ArabicText']
)
save_documents(quran_docs, "original_quran_docs.pkl")

# Hadith
all_hadiths_clean = pd.read_csv(r'Hadith\all_hadiths_clean.csv')
hadith_docs = dataframe_to_documents(
    all_hadiths_clean,
    page_content_column='text_en',
    meta_data_columns=['source', 'chapter_no', 'hadith_no', 'chapter', 'chain_indx', 'text_ar']
)
save_documents(hadith_docs, "original_hadith_docs.pkl")