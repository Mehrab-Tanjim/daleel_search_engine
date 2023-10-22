# Daleel Search Engine

#### Objective: 
Belief in the Quranic verses and authenticated sayings of the Prophet (peace be upon him), collectively known as Hadith, is a religious obligation. Moreover, it is incumbent upon every Muslim to seek education about the fundamental aspects of their religion through the Quran and Hadith. The primary aim of this search engine is to facilitate the convenient retrieval of relevant Quranic Ayats or Hadiths in response to queries or questions, thereby making religious education accessible and user-friendly.

#### Technical Details:
Every ayat and hadith has been transformed into sentence embeddings through the "all-mpnet-base-v2" model. Subsequently, these embeddings are stored in a vector database via the FAISS library. When conducting a search, the query is converted into a sentence embedding using the same model. The MMR algorithm is employed to present search results, emphasizing diversity in the displayed outcomes.

#### Improvment Plan:
This is an initial version (v0) of a basic search engine, and there is significant room for improvement. For instance, consider the query, "What questions will be asked on the Day of Judgment?" It is expected that relevant results, such as Ayat from Surah 102, Verse 7 ("Then, on that Day, you will definitely be questioned about your worldly pleasures"), should be prominently displayed among the top 25 search results. However, the current system does not prioritize these results effectively.

Below are some key observations and proposed improvements:

1. **Data Cleansing and Collection**: In this effort, four datasets were inspected manually: [The Quran Dataset](https://www.kaggle.com/datasets/imrankhan197/the-quran-dataset/), [Holy Quran Dataset](https://www.kaggle.com/datasets/uzairadamjee/holy-quran-dataset), [Sunnah GPT](https://drive.google.com/drive/folders/1UW9Spm7_lVBuV8GMsG4LYwmZPsrHKfh2?usp=sharing), [Clean Hadith Dataset](https://www.kaggle.com/datasets/fahd09/hadith-dataset). `Holy Quran Dataset` and  `Clean Hadith Dataset` are found to be relatively clean. However, issues like malformed sentences, fragmented words, punctuation misuse, and verbosity persist. Data cleansing is crucial, especially for hadiths. In case, these data cannot be further cleaned, an effort is necessary to crawl through good sources and collect high-quality dataset.

2. **Evaluation Criteria**: Developing an evaluation plan is vital for comparing and measuring enhancements. One approach is to involve domain experts, ask specific questions, and assess the top-10 performance of relevant Quranic Ayats and Hadiths.

3. **Preprocessing**: The current system lacks advanced preprocessing techniques, which can enhance search results. Options include stemming, stop word removal, and punctuation removal.

4. **Multiple Translations and Tafseers**: Expanding search to multiple translations and tafseers can enrich the search experience.

5. **Text Chunking**: Breaking down lengthy texts or addressing overlapping sentences can improve content identification.

6. **Text Summarization**: Integrating advanced text summarization techniques, such as GPT-2, Pegasus, T5, Flan-T5, and BART, can facilitate efficient understanding of lengthy ayats and hadith.

7. **Enhanced Search Methods**: Given the evolving embedding models, diverse search algorithms should be considered beyond the standard MMR approach.