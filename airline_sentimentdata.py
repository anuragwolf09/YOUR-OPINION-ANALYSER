#airline sentiment data from kaggle , you can also get this data from kaggle.
import pandas as pd
us_airline_data=pd.read_csv(r'C:\Users\hp\Documents\0000000000002747_training_twitter_x_y_train.csv')
us_airline_data.columns
"""output:Index(['tweet_id', 'airline_sentiment', 'airline', 'airline_sentiment_gold',
       'name', 'negativereason_gold', 'retweet_count', 'text', 'tweet_coord',
       'tweet_created', 'tweet_location', 'user_timezone'],
      dtype='object')"""
airline_data=us_airline_data[['text','airline_sentiment']]
#changing neutral sentiment into positive sentiment:
airline_data['airline_sentiment'] = airline_data['airline_sentiment'].replace(['neutral'], 'Positive')
airline_data['airline_sentiment'] = airline_data['airline_sentiment'].replace(['positive'], 'Positive')
airline_data['airline_sentiment'] = airline_data['airline_sentiment'].replace(['negative'], 'Negative')
#therefore our all three datasets are ready.


#LOADED THIS DATA IN CSV FILE FORMAT FOR DIRECT USE.

#now we have fetched and set all three datasets , now we will merge them.
#head to merging.py
