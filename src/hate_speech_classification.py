"""
Hate speech classification baseline using sklearn
Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
"""

__author__ = "don.tuggener@zhaw.ch"

import csv
import pdb
import pickle
import random
import re
import string
import sys
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

random.seed(42)
STOPWORDS = stopwords.words('english')
STEMMER = SnowballStemmer("english")

translator = str.maketrans('','',string.punctuation)

def read_data(remove_stopwords=True, remove_numbers=True, do_stem=True, remove_punctuation=True, reprocess=False):
	"""
	Read CSV with annotated data.
	We'll binarize the classification, i.e. subsume all hate speach related classes
	'toxic, severe_toxic, obscene, threat, insult, identity_hate'
	into one.
	"""
	if reprocess:
		X, Y = [], []
		for i, row in enumerate(csv.reader(open('data/train.csv', encoding="utf8"))):
			if i > 0:  # Skip the header line
				sys.stderr.write('\r'+str(i))
				sys.stderr.flush()
				text = re.findall('\w+', row[1].lower())
				if remove_stopwords:
					text = [w for w in text if not w in STOPWORDS]
				if remove_numbers:
					text = [w for w in text if not re.sub('\'\.,', '', w).isdigit()]
				if do_stem:
					text = [STEMMER.stem(w) for w in text]
				if remove_punctuation:
					text = [w.translate(translator) for w in text]
				label = 1 if '1' in row[2:] else 0  # Any hate speach label
				X.append(' '.join(text))
				Y.append(label)
		sys.stderr.write('\n')
		pickle.dump(X, open('data/X.pkl', 'wb'))
		pickle.dump(Y, open('data/Y.pkl', 'wb'))
	else:
		X = pickle.load(open('data/X.pkl', 'rb'))
		Y = pickle.load(open('data/Y.pkl', 'rb'))
	print(len(X), 'data points read')
	print('Label distribution:', Counter(Y))
	print('As percentages:')
	for label, count_ in Counter(Y).items():
		print(label, ':', round(100*(count_/len(X)), 2))
	return X, Y


def vectorize(data, method='tfidf'):
	if method == 'tfidf':
		print('Vectorizing with TFIDF', file=sys.stderr)
		tfidfizer = TfidfVectorizer(analyzer='word',)
		features = tfidfizer.fit_transform(data)
		pickle.dump(tfidfizer, open('data/tfidf_vectorizer.pk1', 'wb'))
		return features
	elif method == 'count':
		print('Vectorizing with CountVectorizer', file=sys.stderr)
		vectorizer = CountVectorizer(analyzer='word',)
		return vectorizer.fit_transform(data)

def classify(method='svc'):
	if method == 'svc':
		print('Classification and evaluation (svc)', file=sys.stderr)
		return SVC(kernel='linear')	# Weight samples inverse to class imbalance
	if method == 'nb':
		print('Classification and evaluation (naive bayes)', file=sys.stderr)
		return MultinomialNB()
	if method == 'mlp':
		print('Classification and evaluation (mlpclassifier)', file=sys.stderr)
		return MLPClassifier()
	if method == 'sgd':
		print('Classification and evaluation (sdgclassifier)', file=sys.stderr)
		return SGDClassifier(alpha=1e-05, n_iter=10, penalty='elasticnet', random_state=42)

if __name__ == '__main__':

	print('Loading data', file=sys.stderr)
	X, Y = read_data()

	features = vectorize(X, method='tfidf')

	print('Data shape:', features.shape)
	do_downsample = False
	if do_downsample:	# Only take 20% of the data
		features, X_, Y, Y_ = train_test_split(features, Y, test_size=0.5, random_state=42, stratify=Y)
		print('Downsampled data shape:', features.shape)


	# Randomly split data into 80% training and 20% testing, preserve class distribution with stratify
	X_train, X_test, Y_train, Y_test = train_test_split(features, Y, test_size=0.2, random_state=42, stratify=Y)

	# SVC
	Cs = [10]
	gammas = [0.1]
	kernels = ['rbf']
	param_grid_svc = {'C': Cs, 'kernel': kernels, 'gamma': gammas}

	# grid_search: count, svc: best params: C = 1, gamma = 0.1, kernel = linear
	# f1: 0: 97 1: 75 tot: 95
	# grid_search: tfidf, svc: best params: C = 10, gamma = 0.1, kernel = rbf
	# f1: 0: 98 1: 75 tot: 95

	# MLPClassifier
	# activations = ['identity', 'logistic', 'tanh', 'relu']
	# solvers = ['lbfgs', 'sgd', 'adam']
	# alphas = [0.0001, 0.00001, 0.000001]
	# learning_rates = ['constant', 'invscaling', 'adaptive']
	# param_grid_mlp = {'activation': activations, 'solver': solvers, 'alpha': alphas, 'learning_rate': learning_rates}

	# grid_search: tfidf, mlp: best params: activation = 1e-05, solver = 10, alpha = elasticnet, learning_rate =
	# f1: 0: , 1: , tot:
	# grid_search: tfidf, mlp: best params: activation = 1e-05, solver = 10, alpha = elasticnet, learning_rate =
	# f1: 0: , 1: , tot:

	# SGDClassifier
	alphas = [0.0001, 0.00001, 0.000001]
	penalties = ['elasticnet']
	n_iters = [30, 40, 50]
	param_grid_sgd = {'alpha': alphas, 'penalty': penalties, 'n_iter': n_iters}

	# grid_search: tfidf, sgd: best params: alpha = 1e-05, n_iter = 10, penalty = elasticnet
	# f1: 0: 98, 1: 76, tot: 95
	# grid_search: count, sgd: best params: alpha = 0.0001, n_iter = 150, penalty = elasticnet
	# f1: 0: 97, 1: 75, tot: 95


	#grid_search = GridSearchCV(SVC(kernel='linear'), param_grid_svc, verbose=10, cv=5, n_jobs=2)
	grid_search = GridSearchCV(SGDClassifier(), param_grid_sgd, verbose=10, cv=5, n_jobs=2)
	#grid_search = GridSearchCV(MLPClassifier(), param_grid_mlp, verbose=10, n_jobs=2)
	clf = grid_search.fit(X_train, Y_train)
	pickle.dump(clf, open('data/sgd_clf.pk1', 'wb'))
	print("--------------------- BEST PARAMS ---------------------")
	print(grid_search.best_params_)
	# clf = classify(method='sgd')
	# clf.fit(X_train, Y_train)

	y_pred = clf.predict(X_test)
	print(classification_report(Y_test, y_pred), file=sys.stderr)
	print(confusion_matrix(Y_test, y_pred.tolist()), file=sys.stderr)


	# Apply cross-validation, create prediction for all data point
	# numcv = 5	# Number of folds
	# print('Using', numcv, 'folds', file=sys.stderr)
	# y_pred = cross_val_predict(clf, features, Y, cv=numcv)
	# print(classification_report(Y, y_pred), file=sys.stderr)
