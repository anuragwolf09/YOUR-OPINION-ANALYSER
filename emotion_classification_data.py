#To improve our model we can use emotion classifiction data from kaggle.
#link:link:https://www.kaggle.com/datasets/anjaneyatripathi/emotion-classification-nlp
#this data consist of various random sentences of people which we generally say during conversation.

#fetching and setting the data:
import pandas as pd
data_emotion=pd.read_csv(r'C:\Users\hp\Documents\emotion-labels-train.csv')
data_emotion.columns
#output:Index(['text', 'label'], dtype='object')
print(set(data_emotion['label']))
#output:{'fear', 'sadness', 'joy', 'anger'}
data_emotion.rename(columns = {'label':'Sentiment','text':'Text'}, inplace = True)
data_emotion['Sentiment'] = data_emotion['Sentiment'].replace(['sadness'], 'Negative')
data_emotion['Sentiment'] = data_emotion['Sentiment'].replace(['fear'], 'Negative')
data_emotion['Sentiment'] = data_emotion['Sentiment'].replace(['anger'], 'Negative')
data_emotion['Sentiment'] = data_emotion['Sentiment'].replace(['joy'], 'Positive')
emotion=np.array(data_emotion)
emotion.shape
#output:(3613, 2)

#to tokenize the data we must pass string:
for i in range(len(emotion)):
    emotion[i][0]=str(emotion[i][0])

#now just like we cleaned and then converted the other data into sklearn suitable data , we can clean and convert it too using same functions.
#now before applying count vec to it first combine the other 3 datasets also:
#1.since those three datasets are already cleaned.
#2.clean emotion dataset also.
#3.combine y class of all 4 datasets and combine X class of all 4 datasets by:

#these two were categories and document text of previous 3 datasets:
categories=[category for document,category in dtrain]
documents_text=[' '.join(document) for document,category in dtrain]
documents_text_for_emotion=documents_text #used by kaggle emotion data
categories_for_emotion=categories

#cleaning the emotion dataset:
emotion_train=[(word_tokenize(doc),category) for doc,category in emotion ]
emotion_train=[(clean_review(document),category) for document,category in emotion_train]
y_emotion_train=[category for document,category in emotion_train]
x_emotion_train=[' '.join(document) for document,category in emotion_train]

#joining cleaned above 3 datasets and emotion dataset:
total_train_data=x_emotion_train+documents_text_for_emotion
total_train_y_data=y_emotion_train+categories_for_emotion
#therefore we joined previous 3 datasets and emotion dataset.

#applying count_vec:
xx_train,xx_test,yy_train,yy_test=tt(total_train_data,total_train_y_data)
count_vec1=cv(max_features=3000,ngram_range=(1,2))
total_train_data=count_vec1.fit_transform(xx_train)
total_test_data=count_vec1.transform(xx_test)

#dataset is provided in csv format.

#now our dataset is ready for sklearn classification.
#Head to final_Classification.py for classification_Result.