OPINION SENTIMENT ANALYSER

THIS IS MY PROJECT ON NATURAL LANGUAGE PROCESSING. 

THIS IS PROJECT ABOUT SENTIMENT ANALYSER GIVEN A SENTENCE. 
IN THIS PROJECT I HAVE USED 4 VERY DIFFERENT DATASETS AND TRAINED OUR MODEL WITH THEM TO SEE HOW WELL OUR MODEL PERFORM FOR A EXTREMELY NEW SENTENCE. 

DATASETS USED:
1. AIRLINE SENTIMENT DATASET.
2. TWITTER SENTIMENT DATASET.
3. CONSUMER COMPLAINT DATASET.
4. EMOTION DATASET USED FOR EMOTIONS RECOGNITION TASK.

WORKING :

FIRSTLY I USED ONLY FIRST THREE DATASETS WHICH ARE : AIRLINE SENTIMENT,TWITTER SENTIMENT AND CONSUMER COMPLAINT DATASET.

1. I FETCHED THEM AND SET THEM ACCORDING TO THE NEED . HERE I FIRST MADE SURE THAT I AM USING ONLY TWO COLUMNS WHICH ARE TEXT AND SENTIMENT. 
YOU CAN SEE ALL THREE DATASETS IN :
airlinesentimentdata.py 
complaint_data.py
twitter_sentimentdata.py 

2. AFTER GETTING ALL THREE DATASETS AND SETTING THEM UP ACCORDING TO THE SUITABLE CONTEXT , I MERGED ALL THREE DATASETS TO MAKE A LARGE DATASET HAVING ALL THESE TEXTS AND SENTIMENTS. USING THIS WE GOT A LARGE DATASET HAVING ALMOST ALL TYPE OF EMOTIONS AND SENTENCES AND ALSO THEIR SENTIMENTS.
YOU CAN SEE MERGING IN:
merging.py

3.AFTER MERGING THEM NOW WE NEED TO CLEAN THE DATA. THEREFORE I CLEANED THE DATA USING NLTK TOOLS. 
YOU CAN SEE DATA CLEANING IN:
Cleaningdata.py

I HAVE PROVIDED CLEANED DATA DIRECTLY SO THAT CAN BE USED DIRECTLY FOR COUNT VECTORIZER.

4.AFTER CLEANING THE DATA NOW WE HEAD TO COUNT VECTORIZER SO THAT WE CAN GET OUR DATA INTO SKLEARN CLASSIFIER SUITABLE FORMAT. 
YOU CAN SEE COUNT VECTORIZER IN:
applying_count_vectorizer.py

5.AFTER APPLYING COUNT VECTORIZER WE HEAD TO SKLEARN CLASSIFIER.
YOU CAN SEE IN:
classifyingdata.py

CLASSIFIER AND ACCURACY WE GOT:

SUPPORT VECTOR CLASSIFIER:0.7169177574709805
RANDOM FOREST:0.9138058779945666
MULTINOMIAL NAIVE BAYES:0.8034082489503581

THEREFORE WE GOT BEST ACCURACY USING RANDOM FOREST.

6.TILL NOW WE USED OUR OWN TEST DATA BUT NOW I USED KAGGLE TWITTER SENTIMENT TEST DATA TO CHECK HOW WELL OUR MODEL PERFORMED ON IT:
YOU CAN SEE IN:
classifying_through_test_data.py

SCORE:

RANDOM FOREST :0.9119119119119119
SUPPORT VECTOR CLASSIFIER:0.7527527527527528
MULTINOMIAL NAIVE BAYES:0.7967967967967968 

CONCLUSION:THEREFORE RANDOM FOREST STILL PERFORMED WELL. THEREFORE OUR MODEL IS WORKING GOOD WITH SCORE OF 91%.

SOME OF THE PREDICITONS ARE:

    1 : text: microsoft pay word function poorly samsungus chromebook 🙄
    predicted: Negative
    real: Negative
 
    2 : text: csgo matchmaking full closet hack 's truly awful game
    predicted: Negative
    real: Negative
 
    3 : text: president slap americans face really commit unlawful act acquittal discover google vanityfair.com/news/2020/02/t…
    predicted: Positive
    real: Positive
    
    4 : text: hi eahelp ’ madeleine mccann cellar past 13 year little sneaky thing escape whilst load fifa point take card ’ use paypal account ’ work help resolve please
    predicted: Negative
    real: Negative
    
    5 : text: thank eamaddennfl new te austin hooper orange brown browns austinhooper18 pic.twitter.com/grg4xzfkon
    predicted: Positive
    real: Positive
    
    6 : text: rocket league sea thieves rainbow six siege🤔 love play three stream best stream twitch rocketleague seaofthieves rainbowsixsiege follow
    predicted: Positive
    real: Positive
    
    7 : text: as still knee-deep assassins creed odyssey way anytime soon lmao
    predicted: Positive
    real: Positive
    
    8 : text: fix jesus please fix world go playstation askplaystation playstationsup treyarch callofduty negative 345 silver wolf error code pic.twitter.com/ziryhrf59q
    predicted: Negative
    real: Negative
    
    9 : text: professional dota 2 scene fuck explode completely welcome get garbage
    predicted: Positive
    real: Positive
    
    10 : text: itching assassinate tccgif assassinscreedblackflag assassinscreed thecapturedcollective pic.twitter.com/vv8mogtcjw
    predicted: Positive
    real: Positive
    
7.NOW I ADD ONE MORE DATASET IN OUR MODEL WHICH IS EMOTIONS DATASET WHICH WILL BE USEFUL TO PREDICT SENTIMENT FOR A GENERAL CONVERSATION. 
YOU CAN SEE FETCHING AND SETTING DATASET IN:
emotion_classification_data.py

8.AFTER MERGING ALL 4 DATASETS , THE FINAL SCORE THAT WE GOT:

RANDOM FOREST : 0.8985767648588586

YOU CAN SEE IN:
final_Classification.py

CONCLUSION: OUR MODEL IS WORKING  WELL TO PREDICT SENTIMENT OF A SENTENCE WHICH CAN BE OF ANY TYPE.

I HAVE ALSO TRIED USING NLTK TO FIND OUR OPTIMAL FEATURES AND THEN TO GET CLASSIFIED USING NLTK INBUILT CLASSIFIER.
YOU CAN SEE IN:
finding_own_feature.py 
nltk_classifier.py



