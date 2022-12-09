#we have now cleaned the data.
#now our approach is we will use sklearn classifier to classify the data instead of nltk inbuilt classifier.
#therefore to classify data in sklearn format , we first must need to get our data in sklearn suitable data format. Therefore we use count vectorizer for this.

#to apply count vectorizer we first must get our data into x lable and y label.
categories=[category for document,category in dtrain]
documents_text=[' '.join(document) for document,category in dtrain]
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.model_selection import train_test_split as tt

#splitting the data:
x_train,x_test,y_train,y_test=tt(documents_text,categories)

#applying count vec:
count_vec=cv(max_features=3000,ngram_range=(1,2))
train_data=count_vec.fit_transform(x_train)
test_data=count_vec.transform(x_test)

#now our dataset is ready for sklearn classifier . Head to classifyingdata.py