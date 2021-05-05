import pandas as pd
messages = pd.read_csv('./SMSSpamCollection.txt', sep = '\t', names = ['label', 'message'])

# Data cleaning and preproceinng
import re
import nltk
#nltk.download()

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(x) for x in review if not x in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review)

# Creating the Bag of word Model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])

y = y.iloc[:, 1].values

# train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training model using Naive bayes Classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)













