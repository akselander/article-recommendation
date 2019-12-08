import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from pickle import dump

articles = pd.read_csv('data/EnglishArticles.csv')
articles = articles.drop([['cleaned_summary', 'bert_cleaned_text']])
print(articles.shape)
print(articles.head())
print(articles.isnull().sum())

classic_train, classic_test = train_test_split(
    articles[['cleaned_text']], test_size=0.33, random_state=33)

# nltk.download('wordnet')
# nltk.download('punkt')
wordnet_lemmatizer = WordNetLemmatizer()


def tokenizer(text):
    tktext = word_tokenize(text)
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in tktext]
    return lemmas


vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(
    1, 2), analyzer='word', max_features=None)

classic_train_vector = vectorizer.fit_transform(classic_train['cleaned'])
classic_test_vector = vectorizer.transform(classic_test['cleaned'])

save_npz("classic/train_vector.npz", classic_train_vector)
save_npz("classic/test_vector.npz", classic_test_vector)
