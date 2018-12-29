import nltk
from nltk.corpus import movie_reviews
import random
from nltk.tokenize import word_tokenize

documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]
random.shuffle(documents)

# word_features = movie_reviews.words()
words = nltk.FreqDist(word.lower() for word in movie_reviews.words())
word_features = list(words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        # features['contains({})'.format(word)] = (word in document_words)
        features[word] = (word in document_words)
    input(features)
    return features


featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)

# print(classifier.classify(document_features(word_tokenize("what an amazing movie, brilliant from start to finish!"))))

