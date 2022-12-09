#till now we used sklearn classifier to train our model and to get sentiment.
#now we will use nltk tool set to find our own features and see how well we are doing:
own_feature_data=emotion_train+dtrain #we only take cleaned spliited word data here not joined
own_feature_train_data=own_feature_data[:60000]
own_feature_test_data=own_feature_data[60000:]

#creating array of all words:
all_words=[]
for doc in own_feature_train_data:
    all_words+=doc[0]


#getting freq distribution:
import nltk
freq=nltk.FreqDist(all_words)
common=freq.most_common(3000)
#therefore we took top 3k most occuring words
features=[i[0] for i in common]


#for each doc in documents we will create a dictionary having features and their value:
def get_feature_dict(words):
    current_features={}
    words_set=set(words)
    for w in features:
        current_features[w]=w in words_set
    #it will return true or false value according to if word in feature or not:
    return current_features


own_feature_training_data=[(get_feature_dict(doc),category) for doc,category in own_feature_train_data]
own_feature_test_data=[(get_feature_dict(doc),category) for doc,category in own_feature_test_data]

#now we will apply nltk inbuilt classifier.
#head to nltk_classifier.py