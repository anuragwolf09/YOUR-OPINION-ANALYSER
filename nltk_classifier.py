from nltk import NaiveBayesClassifier as nb
classifier=nb.train(own_feature_training_data)
nltk.classify.accuracy(classifier,own_feature_test_data)
#output:0.6760460293579473
#therefore we got not so good accuracy through nltk inbuilt naive bayes classifier.
#therefore sklearn classifier is working much better.
