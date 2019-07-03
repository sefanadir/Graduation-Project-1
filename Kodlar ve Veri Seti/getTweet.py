# -*- coding: utf-8 -*-
import tweepy
import json,codecs
import sys
import os
import datetime

consumer_key = "..................."
consumer_secret = "............"
access_key = "............"
access_secret = ".........."
try: 
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	auth.get_authorization_url()
	api = tweepy.API(auth)
except tweepy.TweepError:
    print ('Hata')

users_set=[ 
"YapayZekaAI",
"TerapiyeGel",
"DennisRitchie66", 
"doganoncel",
"UgurrcanCoban",
"Osman_gunn",
"aydanszade",
"psiminerva",
"AlitalipSever",
"Sinematopya",
"nonlineernur", 
"CMYLMZ", 
]
with open('kullanıcı_adı_veri_seti.json', 'a' ) as outfile:
	outfile.write('{')
	outfile.write('\n')


for user in users_set:
	public_tweets = api.user_timeline(screen_name = user,count = 100) 


	print(user)
	with open('kullanıcı_adı_veri_seti.json', 'a' ) as outfile:
		outfile.write('"')
		outfile.write('')
		outfile.write(user)
		outfile.write('"')
		outfile.write(':')
		outfile.write(' ')
		outfile.write('[')
	comma_control=0
	for tweet in public_tweets:
			print(tweet.text)
			print(tweet.created_at.isocalendar()[1])
			print('\n')
			
			with open('kullanıcı_adı_veri_seti.json', 'a' ) as outfile:
				if comma_control!=0:
					outfile.write(',')
				comma_control=comma_control+1;
				
				json.dump(tweet.text, codecs.getwriter('utf-8')(outfile), ensure_ascii=False)

	with open('kullanıcı_adı_veri_seti.json', 'a' ) as outfile:
		outfile.write(']')
		outfile.write(',')
		outfile.write('\n')


with open('kullanıcı_adı_veri_seti.json', 'a' ) as outfile:
	outfile.write('}')