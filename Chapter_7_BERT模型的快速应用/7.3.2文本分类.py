from transformers import *

nlp_sentence_classif = pipeline('sentiment-analysis')
print(nlp_sentence_classif(
    'Im not going to lie and say I dont watch the show--I do BUT it has a lot and a lot of flaws 1 The Boarding School is perfect The drama is at a minimum Everyone is so nice to each other you know Lets give that a reality check Its IMPOSSIBLE that ANY school is perfect like PCA Free laptops for everyone Big dorm rooms Mini fridges If there was a school like that in real life almost nobody there would be a virgin for one Two everyone there is so rich and its weird how nobody has anything stolen yet 2 Characters really unrealistic First things first who in theyre right minds talk like they do They talk like a perfect teenager would Secondly Logan ReeseMatthew Underwood is an extremely rich boy hot teenage boy My question is why isnt almost ever girl in that school all over him? Hes rich and hot now a days all those girls would be after him even if he was a jerk Also Chase is the most stupidest person ever He is this shy teenager who claims to not be in love with Zoey and over-reacts to everything that involves Zoey She must be BLIND not to see him in love with her  Come on Nick I know you can do better than THAT Please'))
