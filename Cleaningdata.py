#therefore till now we have collected the data and merged it. 
#now we will clean the data so that we can apply NLP to it:

#getting the data into tokenize form:
from nltk.tokenize import sent_tokenize,word_tokenize
for i in range(len(train_data)):
    train_data[i][0]=str(train_data[i][0])
for i in range(len(train_data)):
    train_data[i][1]=str(train_data[i][1])
dtrain=[(word_tokenize(doc),category) for doc,category in train_data ]
print(dtrain[0])
#output:(['@', 'SouthwestAir', 'I', 'am', 'scheduled', 'for', 'the', 'morning', ',', '2', 'days', 'after', 'the', 'fact', ',', 'yes..not', 'sure', 'why', 'my', 'evening', 'flight', 'was', 'the', 'only', 'one', 'Cancelled', 'Flightled'], 'Negative')

#getting all stopwords and punctuations:
from nltk.corpus import stopwords
import string
stops=set(stopwords.words('english'))
punctuations=string.punctuation
stops.update(punctuations)

from nltk import pos_tag

#defining function which will convert pos_tag parts of speech into wordnet part of speech:
from nltk import WordNetLemmatizer as wnl
from nltk.corpus import wordnet
lemmatizer=wnl()
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

#defining function to clean documents:
def clean_review(words):
    output_words=[]
    for w in words:
        if w.lower() not in stops:
            pos=pos_tag([w])
            clean_word=lemmatizer.lemmatize(w,pos=get_simple_pos(pos[0][1]))
            output_words.append(clean_word.lower())
    return output_words



dtrain=[(clean_review(document),category) for document,category in dtrain]


#now in dtrain we will get cleaned data where stopwords and punctuations will be removed.

#therefore we have cleaned our data now we will apply count vectorizer to it.
#head to applying_count_vectorizer.py