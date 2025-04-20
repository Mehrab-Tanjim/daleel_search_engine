# %% [markdown]
# [IslamQA Scrapper on Colab.](https://colab.research.google.com/drive/1ki_qb7trIKgkPKro_F79zmq2mfdRXKF8?authuser=1#scrollTo=zYbvX59UTZhM)

# %%
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
# %%
df = pd.DataFrame(columns = ['id', 'url', 'title', 'question', 'answer_summary', 'full_answer'])
df

# %%
# Set the base URL and the current range of topic IDs to check
base_url = 'https://islamqa.info/en/categories/topics/'
topic_range = range(270, 400) # 269 topics in 27/04/2023 
topic_ids = []

# Loop through the range of topic IDs and check the HTTP status
for topic_id in topic_range:
  url = base_url + str(topic_id)
  response = requests.get(url)
  if response.status_code != 200: 
    print(f'Topic ID: {topic_id} - Status: {response.status_code}')
  else: 
    topic_ids.append(topic_id)

# %%
# Set the base URL and the range of topic IDs to check
base_url = 'https://islamqa.info/en/categories/topics/'
topic_range = range(1, 270)
topic_question_ids = []

# Loop through the range of topic IDs and scrape the id_number for each question card on the topic
for topic_id in topic_range:
    url = base_url + str(topic_id)
    response = requests.get(url)
    if response.status_code != 200:
      continue
      
    soup = BeautifulSoup(response.content, 'html.parser')
    question_cards = soup.find_all('p', {'class': 'font-number'})

    # specify the question id
    question_range = range(0, len(question_cards), 4)
    questions = [question_cards[i] for i in question_range]
    id_numbers = [card.text.strip() for card in questions]
    topic_question_ids.append(id_numbers)

     # handling the pagination
    next_page = soup.find('li', {'class': 'pagination-link next'})
    if next_page is None:
      print(f'Topic ID: {topic_id} - Number of Questions: {len(topic_question_ids[-1])}')
      continue

    next_page = next_page.find('a')
    next_link = next_page.get('href')
    while next_link is not None:
      url = next_link
      response = requests.get(url)
      soup = BeautifulSoup(response.content, 'html.parser')
      question_cards = soup.find_all('p', {'class': 'font-number'})

      # specify the question id
      question_range = range(0, len(question_cards), 4)
      questions = [question_cards[i] for i in question_range]
      id_numbers = [card.text.strip() for card in questions]
      topic_question_ids[-1].append(id_numbers)

      next_page = soup.find('li', {'class': 'pagination-link next'})
      if next_page is None:
        break
      next_page = next_page.find('a')
      next_link = next_page.get('href')
    
    print(f'Topic ID: {topic_id} - Number of Questions: {len(topic_question_ids[-1])}')

sum([len(x) for x in topic_question_ids]) # before flattening

# %%
# Flatten the current list of lists
def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                if type(item) is list:
                    for sub_item in item:
                        flat_list.append(sub_item)
                else:
                  flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

question_ids = flatten_list(topic_question_ids)

# Remove any duplicates in questions
print(f'Before: {len(question_ids)}')
question_ids.sort()
question_ids = set(question_ids)
question_ids = list(question_ids)
print(f'After: {len(question_ids)}')

# %%
def get_text_strip(soup):
  return soup.text.strip() if soup else soup

# %%
base_url = 'https://islamqa.info/en/answers/'
idx = 0
error_logs = []

# Loop through the range of topic IDs and scrape the id_number for each question card on the topic
for id in tqdm(range(len(question_ids))):
  if id % 500 == 0:
    print(f'Extracted {id} questions..')

  url = base_url + str(id)
  response = requests.get(url)
  if response.status_code != 200:
    error_logs.append(id)
    continue

  soup = BeautifulSoup(response.content, 'html.parser')
  title = get_text_strip(soup.find('h1', {'class':'title is-4 is-size-5-touch'}))
  question = get_text_strip(soup.find(attrs = {'class': 'single_fatwa__question text-justified'}))
  full_answer = get_text_strip(soup.find(attrs = {'class': 'content'}))
  answer_summary = get_text_strip(soup.find(attrs = {'class': 'single_fatwa__summary__body'}))

  df.loc[idx] = [id, url, title, question, answer_summary, full_answer]
  idx += 1
  # save the current df to avoid losing data
  df.to_csv('islamqa_dataset_270_400.csv', index=False)


print(f'Completed {len(question_ids)} questions successfully :D')

# %%
print(f"Number of question_ids to extract: {len(question_ids)}")
print(f"Number of bad question_ids: {len(error_logs)}")
print(f"Number of good question_ids: {len(question_ids) - len(error_logs)}")
print(f"Bad question_ids: {error_logs}")

# %%
df.head()

# %%
df.info()

# %%
'IslamQA scrapped Successfully. Now, convert the df to whatever you want to train with and download it.'

# %%
excel_filename = 'islamqascprapping_270_400.xlsx'
df.to_excel(excel_filename)

csv_filename = 'islamqascprappings_270_400.csv'
df.to_csv(csv_filename)


