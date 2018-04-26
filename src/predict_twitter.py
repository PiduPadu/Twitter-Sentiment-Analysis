import pickle
import random
import re
import string
import sys
import os
import subprocess
import json
from time import sleep

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

classifier_name = 'sgd_clf.pk1'
vectorizer_name = 'tfidf_vectorizer.pk1'
data_folder = 'data/'

subject = str(sys.argv[1])

p = subprocess.Popen(['scrapy', 'crawl', 'TweetScraper', '-a', 'query="from:'+subject+'"'], cwd="D:\CloudStation\Studium\Semester6\KI2\KI2_Lab2_FS2018\TweetScraper\TweetScraper")
sleep(60)
p.terminate()

tweets = []
folder = 'TweetScraper/TweetScraper/Data/tweet'
for file in os.listdir(folder):
    data = json.load(open(folder + file))
    tweets.append(data["text"])

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

random.seed(42)
STOPWORDS = stopwords.words('english')
STEMMER = SnowballStemmer("english")

translator = str.maketrans('','',string.punctuation)

clf = pickle.load(open(data_folder + classifier_name, 'rb'))
vectorizer = pickle.load(open(data_folder + vectorizer_name, 'rb'))

remove_stopwords = True
remove_numbers = True
do_stem = True
remove_punctuation = True

tweetcount = 0
X = []

for i, tweet in enumerate(tweets):
        if  i <= 100000:
            sys.stderr.flush()
            text = re.findall('\w+', tweet.lower())
            if remove_stopwords:
                text = [w for w in text if not w in STOPWORDS]
            if remove_numbers:
                text = [w for w in text if not re.sub('\'\.,', '', w).isdigit()]
            if do_stem:
                text = [STEMMER.stem(w) for w in text]
            if remove_punctuation:
                text = [w.translate(translator) for w in text]
            X.append(' '.join(text))
            tweetcount = tweetcount+1

features = vectorizer.transform(X)
pred = clf.predict(features)
# for i,ele in enumerate(pred):
#     print("{}: {}".format(X[i],ele))
print('\tPercentage negative: {0:.4f}%'.format(sum(pred)/tweetcount*100))
print('\t{} out of {} are negative'.format(pred.tolist().count(1), len(pred)))
