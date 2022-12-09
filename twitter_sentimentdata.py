#collected twitter sentiment data from kaggle
#link:https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
#fetching and setting the data:
import pandas as pd
twitter_data=pd.read_csv(r'C:\Users\hp\Documents\twitter_training.csv')
twitter_data.columns
""" output:Index(['2401', 'Borderlands', 'Positive',
       'im getting on borderlands and i will murder you all ,'],
      dtype='object')"""
#therefore we will rename the columns:
twitter_data.rename(columns = {'Positive':'Sentiment','im getting on borderlands and i will murder you all ,':'Tweets','2401':'index'}, inplace = True)
twitter_data=twitter_data.drop(['index','Borderlands'],axis=1)
print(set(twitter_data.Sentiment))
#output:{'Negative', 'Irrelevant', 'Neutral', 'Positive'}
#now our approach is we will drop Neutral and Irrelevant sentiments and replace them with Positive sentiment.
twitter_data['Sentiment'] = twitter_data['Sentiment'].replace(['Irrelevant'], 'Positive')
twitter_data['Sentiment'] = twitter_data['Sentiment'].replace(['Neutral'], 'Positive')
twitter_data=twitter_data[['Tweets','Sentiment']]
#our this dataset is also ready.

#csv file of this dataset is provided.

#head to airline_sentimentdata.py for next data fetching and setting.