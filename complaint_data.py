#collected consumer complaint data from kaggle
#link:https://www.kaggle.com/selener/consumer-complaint-database
#download the data form given above link.
#fetching and setting our data :
import pandas as pd
consumer_complain_data=pd.read_csv(r'C:\Users\hp\Documents\consumer_complaints_data.csv')
consumer_complain_data.columns
"""output:Index(['Date received', 'Product', 'Sub-product', 'Issue', 'Sub-issue',
       'Consumer complaint narrative', 'Company public response', 'Company',
       'State', 'ZIP code', 'Tags', 'Consumer consent provided?',
       'Submitted via', 'Date sent to company', 'Company response to consumer',
       'Timely response?', 'Consumer disputed?', 'Complaint ID'],
      dtype='object')"""
#we only need Issue and sub-issue column.
consumer_data=consumer_complain_data.drop(['Date received','Product', 'Sub-product', 'Issue', 'Sub-issue',
       'Consumer complaint narrative', 'Company public response', 'Company',
       'State', 'ZIP code', 'Tags', 'Consumer consent provided?',
       'Submitted via', 'Date sent to company', 'Company response to consumer',
       'Timely response?', 'Consumer disputed?', 'Complaint ID'],axis=1)
consumer_complain_data['final_issue']=consumer_complain_data['Issue']+consumer_complain_data['Sub-issue']
#giving each text a sentiment , since all are complaints therefore giving each of them 'negative' sentiment.
sentiment='Negative'
consumer_data['sentiment']=sentiment
consumer_data[0:5]
#hence our first dataset is ready.

#Dataset is already provided therefore use from there instead of getting from kaggle.

#head to twitter_sentiment.py for next data procedure.