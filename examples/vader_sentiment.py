"""
"""
#Enable us to import from directory above:
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..'))

from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
from vader_lexicon import loadLexicon


def get_lexicon():
    with open('../lexicon.pickle', 'rb') as rf:
        lexicon = pickle.load(rf)
    return lexicon

def calculate_sentiment(review):
    lexicon = get_lexicon()

    factors = []
    for sentence in sent_tokenize(review):
        sentFactors = []
        for word in word_tokenize(sentence):
            if word in lexicon:
                sentFactors.append(lexicon[word])
        #Do some maths to calculate score.
        print(sentFactors)

"""
-would be good to store the word and score somewhere to view
-need to determine how to calculate the average of all 'factors'
-rename 'factors' to something clearer

-provide a bank of steam reviews
"""





if __name__ == "__main__":
    review = "Always Sunny is off to a strong start now that it's proven that it can still pull off twists after more than a decade on the air."
    review2 = "what a great game, it's inspiring how much they crammed into this gem! I can't wait to enjoy the sequel."

    calculate_sentiment(review2)