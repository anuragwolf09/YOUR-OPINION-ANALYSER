#now our train and test dataset is ready with all 4 datatsets:

#applying random forest:
random_forest1=rf()
random_forest1.fit(total_train_data,yy_train)

random_forest1.score(total_test_data,yy_test)
#output:0.8985767648588586

#we got a nice accuracy after adding emotion dataset also.

#to test a sentence:
def test1(try_tweet):
    try_tweet=[(word_tokenize(doc)) for doc in try_tweet ]
    try_tweet=[(clean_review(document)) for document in try_tweet]
    try_tweet=[' '.join(document) for document in try_tweet]
    try_tweet=count_vec1.transform(try_tweet)
    predict=random_forest1.predict(try_tweet)
    return predict[0]

#Till now we have used sklearn classifier to classify the sentiment. But now we will create our own features and then we will classify using them.
#head to finding_own_feature.py