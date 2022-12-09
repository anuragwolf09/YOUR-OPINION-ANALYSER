#MERGING:

#till now we have collected all 3 dataset of ours , now we will merge them into one big dataset which we will use then to perform nlp.
#therefore first we will make column name of all three data same:
airline_data.rename(columns={'text':'Text','airline_sentiment':'Sentiment'},inplace=True)
twitter_data.rename(columns={'Tweets':'Text'},inplace=True)
consumer_data.rename(columns={'final_issue':'Text'},inplace=True)
print(airline_data.shape)
print(twitter_data.shape)
print(consumer_data.shape)
"""output:(10980, 2)
          (74681, 2)
          (1282355, 2)"""


import numpy as np
#converting all three datasets into numpy array:
airline_dataset=np.array(airline_data)
twitter_dataset=np.array(twitter_data)
consumer_dataset=np.array(consumer_data)

twitter_dataset=twitter_dataset[:60000]#reduced size by 10k points
#now our dataset look like this:
twitter_dataset[24]
"""array(['The biggest disappointment of my life came a year ago.',
       'Negative'], dtype=object"""
consumer_dataset=consumer_dataset[:10000]#reduced the size to 10k points since all points are negative.

#concatenating all three datasets:
data=np.concatenate((airline_dataset,twitter_dataset,consumer_dataset),axis=0)
data.shape
#output:(80980, 2)
train_data=data

#you can shuffle data it would be better. 


#now we have merged our dataset.
#now we will clean our dataset.
#head to cleaningdata.py