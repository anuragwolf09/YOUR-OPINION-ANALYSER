#function which can take sentence and return the sentiment:
#here the sentence will be given to our trained random forest model .

def test(try_tweet):
    try_tweet=[(word_tokenize(doc)) for doc in try_tweet ]
    try_tweet=[(clean_review(document)) for document in try_tweet]
    try_tweet=[' '.join(document) for document in try_tweet]
    try_tweet=count_vec.transform(try_tweet)
    predict=random_forest.predict(try_tweet)
    return predict[0]

#trying with 10 kaggle test data sentence:
for i in range(10):
    print(sentence_x_kaggle_test[i],end='\n')
    print(test(sentence_x_kaggle_test[i]),end='\n')
    print('real:',y_kaggle_test[i],end='\n')
    print(' ')


"""OUTPUT:
bbc news amazon bos jeff bezos reject claim company act like 'drug dealer bbc.co.uk/news/av/busine…
Positive
real: Positive
 
microsoft pay word function poorly samsungus chromebook 🙄
Positive
real: Negative
 
csgo matchmaking full closet hack 's truly awful game
Positive
real: Negative
 
president slap americans face really commit unlawful act acquittal discover google vanityfair.com/news/2020/02/t…
Positive
real: Positive
 
hi eahelp ’ madeleine mccann cellar past 13 year little sneaky thing escape whilst load fifa point take card ’ use paypal account ’ work help resolve please
Positive
real: Negative
 
thank eamaddennfl new te austin hooper orange brown browns austinhooper18 pic.twitter.com/grg4xzfkon
Positive
real: Positive
 
rocket league sea thieves rainbow six siege🤔 love play three stream best stream twitch rocketleague seaofthieves rainbowsixsiege follow
Positive
real: Positive
 
as still knee-deep assassins creed odyssey way anytime soon lmao
Positive
real: Positive
 
fix jesus please fix world go playstation askplaystation playstationsup treyarch callofduty negative 345 silver wolf error code pic.twitter.com/ziryhrf59q
Positive
real: Negative
 
professional dota 2 scene fuck explode completely welcome get garbage
Positive
real: Positive
 """

#therefore we got pretty good accuracy till now.Now we will add a whole new one more dataset and see if our model imporve or not.
#head to emotion_classification_data.py