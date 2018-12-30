# import nltk
# from nltk.corpus import movie_reviews
# import random
# from nltk.tokenize import word_tokenize

# documents = [
#     (list(movie_reviews.words(fileid)), category)
#     for category in movie_reviews.categories()
#     for fileid in movie_reviews.fileids(category)
# ]
# random.shuffle(documents)
# input(documents)
# input(type(documents))

# # word_features = movie_reviews.words()
# words = nltk.FreqDist(word.lower() for word in movie_reviews.words())
# word_features = list(words)[:2000]

# def document_features(document):
#     document_words = set(document)
#     features = {}
#     for word in word_features:
#         # features['contains({})'.format(word)] = (word in document_words)
#         features[word] = (word in document_words)
#     # input(features)
#     return features


# featuresets = [(document_features(d), c) for (d,c) in documents]
# train_set, test_set = featuresets[100:], featuresets[:100]
# classifier = nltk.NaiveBayesClassifier.train(train_set)

# # print(nltk.classify.accuracy(classifier, test_set))
# classifier.show_most_informative_features(5)


# print(classifier.classify(document_features("""A thoroughly entertaining ride.""")))

# # print(classifier.classify(document_features(word_tokenize("what an amazing movie, brilliant from start to finish!"))))




import os
dataPath = os.path.join(__file__,'..','..','..','data','steam_reviews')
posData = os.path.join(dataPath,'pos.txt')
negData = os.path.join(dataPath,'neg.txt')
from collections import Counter
from nltk.tokenize import word_tokenize
import random

#Create a list of stopwords:
from nltk.corpus import stopwords
import string
stopWords = list(set(stopwords.words('english')))
stopWords.extend(string.punctuation)


def clean_reviews(reviews):
    """
    """
    reviews = [review.lower() for review in reviews] #Lower-case text.
    reviews = [review.strip() for review in reviews] #Remove '\n', etc.
    return reviews

def generate_vocabulary(posPath, negPath, featureSize=2500):
    """
    """
    reviews = [] #To contain ALL reviews.
    #Load positive review data:
    with open(posPath, 'r', encoding='utf-8') as rf:
        reviews.extend(rf.readlines())
    #Load negative review data:
    with open(negPath, 'r', encoding='utf-8') as rf:
        reviews.extend(rf.readlines())

    #Clean reviews - ensures 'smarter' vocabulary:
    reviews = clean_reviews(reviews)

    #Instantiate a word counter and count occurrences of words:
    wordCounter = Counter()
    for review in reviews:
        wordCounter.update(word_tokenize(review))

    #Remove stopwords from our counter:
    for word in stopWords:
        if word in wordCounter:
            del wordCounter[word]

    print("No. Unique Words={}".format(len(wordCounter)))
    print("Most common words:\n{}".format(wordCounter.most_common(10)))
    wordFeatures = wordCounter.most_common(featureSize)

    return wordFeatures

def format_documents(posPath, negPath):
    """
    """
    documents = [] #To contain document data.
    #Load positive review data:
    with open(posPath, 'r', encoding='utf-8') as rf:
        reviews = [(word_tokenize(review), 'pos') for review in rf.readlines()]
        documents.extend(reviews)
    #Load negative review data:
    with open(negPath, 'r', encoding='utf-8') as rf:
        reviews = [(word_tokenize(review), 'neg') for review in rf.readlines()]
        documents.extend(reviews)

    random.shuffle(documents)
    return documents


if __name__ == "__main__":
    print(format_documents(posData, negData))
    # generate_vocabulary(posData, negData)
