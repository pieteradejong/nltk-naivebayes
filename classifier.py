import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

def word_features(words):
  return dict([(word, True) for word in words])

neg_ids = movie_reviews.fileids('neg')
pos_ids = movie_reviews.fileids('pos')

neg_features = [(word_features(movie_reviews.words(fileids=[f])), 'neg') for f in neg_ids]
pos_features = [(word_features(movie_reviews.words(fileids=[f])), 'pos') for f in pos_ids]

neg_cutoff = len(neg_features)*3/4
pos_cutoff = len(pos_features)*3/4

training_features = neg_features[:neg_cutoff] + pos_features[:pos_cutoff]
test_features = neg_features[neg_cutoff:] + pos_features[pos_cutoff:]

print 'Training on %d instances, testing on %d instances' % (len(training_features), len(test_features))

classifier = NaiveBayesClassifier.train(training_features)
print 'Accuracy: ', nltk.classify.util.accuracy(classifier, test_features)
classifier.show_most_informative_features()
