import pandas as pd
from nltk.corpus import stopwords
from pickle import dump

articles = pd.read_csv('data/EnglishArticles.csv')
print(articles.shape)
print(articles.head())
print(articles.isnull().sum())

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
}

# nltk.download('stopwords')


def clean_text(text, remove_stopwords=True, remove_interpunction=True):
    import re

    text = text.lower()

    text = text.split()
    new_text = []

    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)

    text = " ".join(new_text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    if remove_interpunction:
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


articles['cleaned_text'] = articles['TEXT'].apply(
    lambda text: clean_text(text))
print("Texts are cleaned for Classic.")

articles['bert_cleaned_text'] = articles['SUMMARY'].apply(
    lambda text: clean_text(text, remove_interpunction=True),)
print("Texts are cleaned for BERT.")

articles['cleaned_summary'] = articles['SUMMARY'].apply(
    lambda text: clean_text(text, remove_stopwords=False, remove_interpunction=False),)
print("Summaries are cleaned.")

articles = articles[['cleaned_text', 'bert_cleaned_text', 'cleaned_summary']]
print(articles.shape)
print(articles.head())
print(articles.isnull().sum())

articles.to_csv('data/CleanedArticles.csv')

stories = list()
for article in articles[['bert_cleaned_text', 'cleaned_summary']].values:
    text, summary = article
    stories.append({'story': text, 'highlights': summary})

# save to file
dump(stories, open('summarization/article_dataset.pkl', 'wb'))
