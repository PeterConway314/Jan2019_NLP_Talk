import pickle
from nltk.tokenize import word_tokenize

def word_feats(words):
    return dict([(word.lower(), True) for word in words])


#Load our classifier from file:
with open('classifier.pickle', 'rb') as rf:
    classifier = pickle.load(rf)

#Check most informative features (sanity check):
classifier.show_most_informative_features(15)

#Test the classifier on some example reviews:
test_review = ""
features = word_feats(word_tokenize(test_review))
classification = classifier.classify(features)
print("{}\n={}\n".format(test_review, classification))
