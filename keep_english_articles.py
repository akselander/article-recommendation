import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from pickle import dump, load

articles = pd.read_csv('data/EveryArticle.csv')
print(articles.shape)
print(articles.head())
print(articles.isnull().sum())

pattern = r'http(s)?:\/\/www\.[A-z0-9]*\.com.*'
nonEnglishCharacter = r'[^\x00-\x7F]+'


articles = articles[articles['URL'].str.contains(pattern)]
articles = articles[articles['TEXT'].str.contains(
    nonEnglishCharacter) == False]
articles = articles[articles['TITLE'].str.contains(
    nonEnglishCharacter) == False]
articles = articles[['TITLE', 'TEXT', 'SUMMARY']]
articles = articles.dropna()
articles = articles.drop_duplicates()
articles = articles.reset_index(drop=True)

print(articles.head())
for i in range(5):
    print("Article #", i+1)
    print(articles.TITLE[i])
    print(articles.SUMMARY[i])
    print(articles.TEXT[i])
    print()

articles.to_csv('data/EnglishArticles.csv')
