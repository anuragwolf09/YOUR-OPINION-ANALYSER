#Till now we classified using our own test data that we splitted from dtrain. 
#now we use test data provided by kaggle in its twitter sentiment analysis dataset.
#you can get the test data from:
##link:kaggle_test_data=pd.read_csv(r'C:\Users\hp\Documents\twitter_validation.csv')
import pandas as pd
import numpy as np
kaggle_test_data=pd.read_csv(r'C:\Users\hp\Documents\twitter_validation.csv')
kaggle_test_data.columns
#changing column names:
kaggle_test_data.rename(columns = {'Irrelevant':'Sentiment','I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣':'Tweets'}, inplace = True)
kaggle_test_data=kaggle_test_data[['Tweets','Sentiment']]
kaggle_test_data['Sentiment'] = kaggle_test_data['Sentiment'].replace(['Neutral'], 'Positive')
kaggle_test_data['Sentiment'] = kaggle_test_data['Sentiment'].replace(['Irrelevant'], 'Positive')
kaggle_test_data.rename(columns={'Tweets':'Text'},inplace=True)
kaggle_test_data=np.array(kaggle_test_data)

#cleaning kaggle test data by the functions we used to clean train data:
kaggle_test=[(word_tokenize(doc),category) for doc,category in kaggle_test_data ]
kaggle_test=[(clean_review(document),category) for document,category in kaggle_test]

y_kaggle_test=[category for document,category in kaggle_test]
sentence_x_kaggle_test=[' '.join(document) for document,category in kaggle_test]
x_kaggle_test=sentence_x_kaggle_test

#applying count vec:
x_kaggle_test=count_vec.transform(x_kaggle_test)

#using random forest to predict:

#we have already trained our random forest model through dtrain previously.
y_kaggle_test_predict=random_forest.predict(x_kaggle_test)
random_forest.score(x_kaggle_test,y_kaggle_test)
#output:0.9119119119119119
print(cr(y_kaggle_test,y_kaggle_test_predict))
"""output:precision    recall  f1-score   support

   Negative       0.83      0.83      0.83       266
   Positive       0.94      0.94      0.94       733

avg / total       0.91      0.91      0.91       999"""

#therefore we again got good score through random forest.

naive_bayes.score(x_kaggle_test,y_kaggle_test)
#ouput:0.7967967967967968
svc.score(x_kaggle_test,y_kaggle_test)
#output:0.7527527527527528

#CONCLUSION:RANDOM FOREST PERFORMED WELL.


#to get first 100 predictions on test data by random forest:

for i in range(1,101):
    print(i,':',end=' ')
    print('text:',sentence_x_kaggle_test[i],end='\n')
    print('predicted:',y_kaggle_test_predict[i],end='\n')
    print('real:',y_kaggle_test[i],end='\n')
    print(' ')
"""
#OUTPUT:
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
 
11 : text: fredtjoseph hey fred comcast cut cable verizon stay call shut pic.twitter.com/cpwsrmuedg
predicted: Negative
real: Negative
 
12 : text: csgo wingman im silver dont bully twitch.tv/lprezh
predicted: Positive
real: Positive
 
13 : text: nba2k game suck ... 2 38 second left team intentionally foul
predicted: Negative
real: Negative
 
14 : text: congrats nvidia nemo team 1.0.0 release candidate really excite see nemo embrace hydra way take control configuration madness machine learn
predicted: Positive
real: Positive
 
15 : text: yeah ’ fun
predicted: Positive
real: Positive
 
16 : text: fuck life 😆
predicted: Negative
real: Negative
 
17 : text: happy birthday red dead redemption shit change life crazy experience
predicted: Positive
real: Positive
 
18 : text: say microsoft hardware software security man get hack
predicted: Negative
real: Negative
 
19 : text: new callofduty ps5 🔥🔥🔥🔥 oh god 😭😍
predicted: Negative
real: Negative
 
20 : text: anyone play bad luck albatross deck hearthstone literal cop fucking fun police pic.twitter.com/jy6trq351e
predicted: Positive
real: Positive
 
21 : text: call duty warzone livestream w/ sub warzone youtu.be/7bhh_pjomu4 via youtube please come watch amazing call duty warzone stream amazing streamer 'd really really nice give view like well 😀 cod callofduty warzone
predicted: Positive
real: Positive
 
22 : text: finally played rainbow six siege first time ... admit prefer pull hair csgo day
predicted: Negative
real: Negative
 
23 : text: umm playapex die say bug pic.twitter.com/bzmhzbadof
predicted: Positive
real: Positive
 
24 : text: gtc20 nice motivational accessible nvidia/ai product fair related tech talk nvidia.com/en-us/gtc/keyn… interest interaction/social activity braindates dinner stranger ... free attendance university reg.rainfocus.com/flow/nvidia/gt…
predicted: Positive
real: Positive
 
25 : text: yo verizon add 120 'fee account covid19 protection without permission force pay check bill carefully
predicted: Negative
real: Negative
 
26 : text: might last team make difficult decision update overwatchleague nyxl overwatch overwatch2 blizzard game lockdown pic.twitter.com/di1htl4mcv
predicted: Negative
real: Positive
 
27 : text: best squad yet pubg pubgmobile pubgkenya instagram.com/p/b-obt_eaa4f/…
predicted: Positive
real: Positive
 
28 : text: borderlands submit complaint ceo n't pay staff bonus
predicted: Negative
real: Negative
 
29 : text: watching nvidia position lead hardware manufacturer also provide meaningful software consumer remarkable thing beauty incredibly lead company clear focus goal well do nvidia
predicted: Negative
real: Positive
 
30 : text: ’ see look like xbox controller ’ say anything anyway fire
predicted: Positive
real: Positive
 
31 : text: johnson johnson knowingly sell baby powder contain asbestos decade tasked produce covid19 vaccine ... along corporate criminal glaxosmithkline receive 2nd large fine corporate history various crime trust 😳
predicted: Negative
real: Negative
 
32 : text: thing would nvidia 3090 ... unspeakable 🧐
predicted: Positive
real: Positive
 
33 : text: fortnite run like ass.. fps drop everywhere wtf
predicted: Negative
real: Negative
 
34 : text: great play dude good optic mk2 carbine 👌
predicted: Positive
real: Positive
 
35 : text: get horse back ps4live red dead redemption 2 live youtu.be/9bvkh67oaei
predicted: Positive
real: Positive
 
36 : text: really disappoint move remedy bought control day one season pas get ps5 upgrade rebuy everything repackaged ultimate edition
predicted: Positive
real: Positive
 
37 : text: finish assassins creed odyssey shadow_official shadow_na begin end thanks amaze game experience service 👏
predicted: Positive
real: Positive
 
38 : text: solo q freak spin fast low fps thought take b rainbow6game http //t.co/mlvhmu4qez
predicted: Positive
real: Positive
 
39 : text: eldest spent long play new call duty game ’ convince even know day let alone long lockdown last ’ normally object think ’ rather deal frustrate stuck home
predicted: Positive
real: Positive
 
40 : text: //the corruption knocking overwatch babes serious mood atm ... pic.twitter.com/qny4kmgblv
predicted: Positive
real: Positive
 
41 : text: playcodmobile m2games2 love call duty unfortunately 'm lot audio bug every time 'm play br mode game audio stop close open dnv device 'm use ios
predicted: Positive
real: Positive
 
42 : text: amazon stop accept new online grocery customer cut hour whole foods store dailymail.co.uk/news/article-8… http //t.co/jn7omalq3n
predicted: Positive
real: Positive
 
43 : text: gon na lie black ops cold war trailer probably best cod teaser ever see
predicted: Positive
real: Positive
 
44 : text: melusi 4k shock justhoneybadger simmyster06 walkerpool mdassassin007 owahid65 rainbowsixsiege pic.twitter.com/0acuqjtvgl
predicted: Positive
real: Positive
 
45 : text: never popped ice block hearthstone pic.twitter.com/bcszlhjhpf
predicted: Negative
real: Negative
 
46 : text: mean johnson johnson suppress report asbestos contamination talc product decade wonder many die birx say `` randomized control study show evidence improve outcome hcq '' liar youtu.be/77tmszubru4
predicted: Positive
real: Negative
 
47 : text: ’ addict call duty mobile😅
predicted: Negative
real: Negative
 
48 : text: could go well ... horribly wrong
predicted: Positive
real: Positive
 
49 : text: yesterday decide go back overwatch ... forgot much love play want play nothing else mean event atm.. get ta get pharah skin anas already 💛
predicted: Positive
real: Positive
 
50 : text: pubg_support hiya 'll u fix f'n console game ask lot satisfied lack response ☺️ wonder lagouts part game like red zone 'm aware ca n't fix 2 year problem thx
predicted: Negative
real: Negative
 
51 : text: fifa leslie steiger must joking see horrible thing ift.tt/2y1tbto ///
predicted: Negative
real: Negative
 
52 : text: thank want stay home play fifa easportsfifa eahelp pic.twitter.com/7vw03aflay
predicted: Positive
real: Positive
 
53 : text: beverlycitizen ronald bellanti resident beverly work ground control organization found drunk drive prevention campaign take facebook express deplorably racist violent view please help expose pic.twitter.com/78ritukem2
predicted: Positive
real: Positive
 
54 : text: milestone anniversary chicopee home depot congratulations ed 29 year company gain another gold milestone tim 14 year company mark gain another silver milestone time fly 're fun pic.twitter.com/dilaa5cufx
predicted: Positive
real: Positive
 
55 : text: like killstreaks
predicted: Positive
real: Positive
 
56 : text: card magnificent nba2kmyteam sfcrabs pinkdiamond sprewell nba2k spree myteam card grinding nba2k20 pic.twitter.com/2iqsf9zqw0
predicted: Negative
real: Positive
 
57 : text: anything wrong game cant claim credit playcodmobile joke ... callofdutymobile codm codmobile callofduty http //t.co/0geogc0egq
predicted: Negative
real: Negative
 
58 : text: ok 'm block man 's new level
predicted: Positive
real: Positive
 
59 : text: take phone number thanks verizon rey pay bill yall could n't jus give time tf
predicted: Negative
real: Negative
 
60 : text: big shout icklenellierose remind great assassins creed black flag never finish stuck home broken car whilst ps4 switch girlfriend doom acnh ps3 gamecube gon na finish pic.twitter.com/mrqdajiyvg
predicted: Positive
real: Positive
 
61 : text: love rainbow6game much 💙
predicted: Positive
real: Positive
 
62 : text: everyone know story true life-long friendship everyone know big chunk life miss end reason big thanks marykhln fill void great vibe conversation
predicted: Positive
real: Positive
 
63 : text: nba2k fuck joke
predicted: Negative
real: Negative
 
64 : text: johnson johnson stop sell baby powder us firm face thousand lawsuit talc product cause cancer say oct test found asbestos baby powder test conduct usfda discover trace amount bbc.co.uk/news/business-…
predicted: Negative
real: Positive
 
65 : text: robodanjal chance could get list map new pubg tdm ca n't seem find good list anywhere
predicted: Negative
real: Negative
 
66 : text: sound enjoy groove little montage make tribute desert eagle powerful badass secondary weapon ever callofduty modernwarfare xbox joececot artpeasant drift0r http //t.co/wooktjkbmz
predicted: Negative
real: Positive
 
67 : text: seems like playstation marketing deal callofdutyblackopscoldwar feels good treat well playstation 🙌
predicted: Positive
real: Positive
 
68 : text: eamaddennfl hi try golden ticket challenge work 10 friend u post point
predicted: Positive
real: Negative
 
69 : text: much enjoy ghostreconbreakpoint division2 love lore title game gamebooks instagram.com/p/b9feinlnn1u/…
predicted: Positive
real: Positive
 
70 : text: 1982 johnson johnson 's extra strength tylenol recall seven people die take haitian music
predicted: Positive
real: Negative
 
71 : text: god 🥺
predicted: Positive
real: Negative
 
72 : text: satyanadella microsoft thanks celebrate diversity need positive energy day
predicted: Positive
real: Positive
 
73 : text: ’ best one wtf
predicted: Positive
real: Positive
 
74 : text: aoc make ignorant comment come wi meet
predicted: Positive
real: Positive
 
75 : text: oooooh shit think motherboard already compatible
predicted: Positive
real: Positive
 
76 : text: nyummm delicious finally good content nyumm skins asher
predicted: Positive
real: Positive
 
77 : text: call duty modern warfare problem child soldier level newsychronicles.com/ p=3309 utm_so…
predicted: Positive
real: Positive
 
78 : text: late b_gardiner 's random musings paper.li/b_gardiner/131… thanks bjoerndarko fmscreative google seo
predicted: Positive
real: Positive
 
79 : text: “ free software movement dead linux ’ exist 2007 even linus get job today ” bill hilf microsoft mainstream medium headline remove article later techrights.org/2007/07/27/bil…
predicted: Positive
real: Positive
 
80 : text: people want play valorant say gon na pursue professionally go play 100 hour csgo still like game think would good game bore mind would recommend pursue
predicted: Positive
real: Positive
 
81 : text: fuckkkkkk cant wait
predicted: Positive
real: Positive
 
82 : text: current status gold make monthly token purchase dont find stable source income bad 3 month mithril flamebloom broke seller warcraft pic.twitter.com/l7xmqxwpz0
predicted: Positive
real: Positive
 
83 : text: absolute funniest interaction ’ ever see league legends pic.twitter.com/nswyumdvrx
predicted: Positive
real: Positive
 
84 : text: use voice changer pretend girl csgo match last night dms flood xd
predicted: Positive
real: Positive
 
85 : text: oh shit get 1 day finish fuk
predicted: Positive
real: Negative
 
86 : text: ca n't find confirmation anywhere look like fenix_app quit mute keywords talon strip imaginary sport timeline may need switch blaseball like fandom pubg least mute filter work pubg
predicted: Positive
real: Positive
 
87 : text: miss hearthstone simpler
predicted: Negative
real: Negative
 
88 : text: gm fam hope great today ... jus want take second thank follower support guy truly awesome💪🏾👍🏾🙌🏾 ... stateofdecay2 ghostreconbreakpoint reddeadredemption2 smallstreamercommunity
predicted: Positive
real: Positive
 
89 : text: play ptitdrogo blyonfire start 19 cest today get chance win awesome prize merch store pl4zmacom gaming booster sounds like good deal join discord discord.gg/nakhtps tomorrow hearthstone rye_viper http //t.co/jkxixjco29
predicted: Positive
real: Positive
 
90 : text: leaked memo excoriates facebook ’ ‘ slapdash haphazard ’ response global political manipulation dlvr.it/rgbzjd http //t.co/m7jamqef7e
predicted: Positive
real: Negative
 
91 : text: look kinda clean
predicted: Positive
real: Positive
 
92 : text: wilson 💛
predicted: Positive
real: Positive
 
93 : text: flip fuck cyklon30001189 join kingdom mixer mixer.com/deduke mixerpartner mixer streamer xbox callofduty
predicted: Positive
real: Positive
 
94 : text: blizzardcs try buy overwatch credit debit load say `` something go wrong select another payment method try later '' help try hour
predicted: Negative
real: Negative
 
95 : text: verizon waive data overage charge tough folk
predicted: Negative
real: Negative
 
96 : text: one buy battlefield 3 steam ’ app literally run google search recommend ’ waste money steamsummersale steamsummersale2020
predicted: Negative
real: Negative
 
97 : text: hisaperth obiawards ceremony take place friday 29 may celebrating outstanding best inspiring work staff student within perthcollegeuhi do year streaming live 12 noon friday hisaperth facebook page 🏆 teamhisa awards iwouldliketothank pic.twitter.com/re3iy9t66p
predicted: Negative
real: Positive
 
98 : text: indigo urgent care look microsoft teams microsoft ’ power platform help deliver quality care world-class patient experience lnkd.in/eazwmub
predicted: Positive
real: Positive
 
99 : text: 🤔 sure data go frustrated kid use immediately 😠 verizon app power hand ⚡ download app store via google play 👍
predicted: Positive
real: Positive
 
100 : text: ghostrecon ghostrecon_uk pvp server n't work ... nobody connect pls delete scope drone 're useless und block syrinx ... bugpoint brokenpoint pic.twitter.com/qktnee6dvz
predicted: Negative
real: Negative
 
"""


#I have added try a sentence function which will take the sentence and use our this model to predict the sentiment.Head to try_a_sentence.py to see that.


#now we have got our result. We Performed pretty well but we will try to add one more dataset and then we will see how good we perform then.
#head to emotion_classification_data.py for our new dataset.
