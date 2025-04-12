# Daleel Search Engine

#### Live!
Please try and leave your feedback: [daleel.streamlit.app](https://daleel.streamlit.app/)
#### Objective: 
Belief in the Quranic verses and authenticated sayings of the Prophet (peace be upon him), collectively known as Hadith, is a religious obligation. Moreover, it is incumbent upon every Muslim to seek education about the fundamental aspects of their religion through the Quran and Hadith. The primary aim of this search engine is to facilitate the convenient retrieval of relevant Quranic Ayats or Hadiths in response to queries or questions, thereby making religious education accessible and user-friendly.

#### Technical Details:
What does the search engine is capable of doing? Given a part of text that we remember to the best of our ability, find the nearest meaning or most similar hadith or quran. 

What does this app is not capable of doing? Given a free form question, given an answer based on the hadith and quran. Or intelligently find what ayats or hadiths might be relevent for answering this. 
Note this is more challenging problem, as not only does it require a semantic matching but also an understanding or reasoning. This is difficult to achieve without using a powerful model, like LLM. However, since the content of the hadiths and quran is static, what we could do is to generate a bunch of questions from hadith/quran and trying to finetune a model based on it. Now, based on this, IslamQA might seem a good source but we will need to clean the dataset. As sometimes, the answerer choose other hadith or ayat to make a point and then finally quote a relevant ayat and hadith. So, there needs to be some cleaning for this dataset. Currently we have this dataset from Kaggle and Huggingface and extracted the hadith and ayats for every question. But this is noisy for the reasons mentioned earlier. That being said, this dataset can be used for comparative analysis, like which of the processing work better for the same model, or which model might work better for open-ended questions.

Every ayat and hadith has been transformed into sentence embeddings through the ["all-mpnet-base-v2" model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). Subsequently, these embeddings are stored in a vector database via the [FAISS library](https://python.langchain.com/docs/integrations/vectorstores/faiss). When conducting a search, the query is converted into a sentence embedding using the same model. The MMR algorithm is employed to present search results, emphasizing diversity in the displayed outcomes.

#### Leaderboard
Currently it seems  "nomic-ai/nomic-embed-text-v2-moe" with proper pretexting (like search_query, search_document) is most competitive based on a quick human study. This model without pretexting seem to perform better for open-ended question (like what questions Allah will ask on the day of Judgment). Then comes "Alibaba-NLP/gte-multilingual-base" which seem to work better for other languanges. 

Moreoever original texts of the Quran (instead of processed) and processed texts of Hadith (instead of original) work better. This is based on the evaluation on IslamQA dataset.

#### Improvment Plan:
For immediate improvement plan, we can use multiple embedding models and do a cross encoding based on the results. Another immediate improvement plan is to use multiple translations for quran.

This is an initial version (v0) of a basic search engine, and there is significant room for improvement. For instance, consider the query, "What questions will be asked on the Day of Judgment?" It is expected that relevant results, such as Ayat from Surah 102, Verse 7 ("Then, on that Day, you will definitely be questioned about your worldly pleasures"), should be prominently displayed among the top 25 search results. However, the current system does not prioritize these results effectively.

Below are some key observations and proposed improvements:

1. **Fine Tuning**: The most important part that can increase the performance by a lot is fine-tuning the model, which can have better matching between text and verses/ayats.
2. **Data Cleansing and Collection**: In this effort, four datasets were inspected manually: [The Quran Dataset](https://www.kaggle.com/datasets/imrankhan197/the-quran-dataset/), [Holy Quran Dataset](https://www.kaggle.com/datasets/uzairadamjee/holy-quran-dataset), [Sunnah GPT](https://drive.google.com/drive/folders/1UW9Spm7_lVBuV8GMsG4LYwmZPsrHKfh2?usp=sharing), [Clean Hadith Dataset](https://www.kaggle.com/datasets/fahd09/hadith-dataset). `Holy Quran Dataset` and  `Clean Hadith Dataset` are found to be relatively clean. However, issues like malformed sentences, fragmented words, punctuation misuse, and verbosity persist. Data cleansing is crucial, especially for hadiths. In case, these data cannot be further cleaned, an effort is necessary to crawl through good sources and collect high-quality dataset.

3. **Evaluation Criteria**: Developing an evaluation plan is vital for comparing and measuring enhancements. One approach is to involve domain experts, ask specific questions, and assess the top-10 performance of relevant Quranic Ayats and Hadiths.

4. **Preprocessing**: The current system lacks advanced preprocessing techniques, which can enhance search results. Options include stemming, stop word removal, and punctuation removal.

5. **Multiple Translations and Tafseers**: Expanding search to multiple [translations](https://huggingface.co/datasets/tarteel-ai/quran-tafsir) and [tafseers](https://www.kaggle.com/code/alizahidraja/quran-nlp/input?select=Quran_English_with_Tafseer.csv) can enrich the search experience.

6. **Text Chunking**: Breaking down lengthy texts in [chunks](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter) or addressing overlapping sentences can improve content identification.

7. **Text Summarization**: Integrating advanced [text summarization techniques](https://huggingface.co/learn/nlp-course/chapter7/5?fw=pt#models-for-text-summarization), such as GPT-2, Pegasus, T5, Flan-T5, and BART, can facilitate efficient understanding of lengthy ayats and hadith.

8. **Enhanced Search Methods**: Given the evolving embedding models, diverse models (e.g., SOTA model from the [leaderboard](https://huggingface.co/spaces/mteb/leaderboard)) and various search algorithms should be considered beyond the standard MMR approach (e.g., reranking models, see point [2] from [here](https://huggingface.co/BAAI/bge-base-en-v1.5#model-list)).
