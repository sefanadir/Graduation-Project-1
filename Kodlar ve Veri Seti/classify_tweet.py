# -*- coding: utf-8 -*-
import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import string
import unicodedata
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)

# a table structure to hold the different punctuation used
tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
    if unicodedata.category(unichr(i)).startswith('P'))


# method to remove punctuations from sentences.
def remove_punctuation(text):
    return text.translate(tbl)

# initialize the stemmer
stemmer = LancasterStemmer()
# variable to hold the Json data read from the file
data = None


# read the json file and load the training data
with open('sunum_set.json') as json_data:
    data = json.load(json_data, encoding='utf-8')

print(data)

# get a list of all categories to train for
categories = list(data.keys())
words = []
# a list of tuples with words in the sentence and category name
docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        # remove any punctuation from the sentence
        each_sentence = remove_punctuation(each_sentence)
        print(each_sentence)
        # extract words from each sentence and append to the word list
        w = nltk.word_tokenize(each_sentence)
        print("tokenized words: ", w)
        words.extend(w)
        docs.append((w, each_category))

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

print(words)
print(docs)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(categories)


for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # our training set will contain a the bag of words model and the output row that tells
    # which catefory that bow belongs to.
    training.append([bow, output_row])

# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)

# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')


# let's test the mdodel for a few sentences:
# the first two sentences are used for training, and the last two sentences are not present in the training data.
tweet_1  = "Kredi kartiyla milli piyango bileti almak " #@zehirli_oksijen
tweet_2  = "EserYenenlerShow bu aksam 23.15'te TV8'de. eseryeneler hayirli olsun Eser'im :) https://t.co/8TjBAfItoL" #@acunilicali
tweet_3  = "ekonomikanaliz arzovaone Hocam basiniz sag olsun. Allah rahmet etsin." #@bmylz
tweet_4  = "muzik Bu sene Grammy'yi 'Rock' kategorisinde bu parca alir: https://t.co/QRnnKX29Sc" #@kaansezyum
tweet_5  = "RT @haluk_levent: Agresif bir sarki bu.. https://t.co/4Ys9ZIsR4L" #@haluk_levent
tweet_6  = "@olsunbenolmus Gecim zorlugu ile nezaket ve iletisim dersi almak arasindaki baglantiyi kuramadim...?" #@TerapiyeGel
tweet_7  = "Yeni mama makineleriyle daha cok sokak hayvanini doyurmak ister misin? #BuMamaBenden desen yeter! @vodafonevakfi" #@glshybs
tweet_8  = "RT @ertugrulbaskan: Gecenlerde mecburi bir mangal partisine katildim.Sesimi cikarmadan koz patlican kollarken niye yemiyorsun dediler.Sessi…" #@anothermorning2
tweet_9  = "Le Notti di Cabiria (1957) https://t.co/taw4XCUYGV" #@Sinematopya
tweet_10 = "bi de bunlarin sadece sofor onunde kucuk delik oyanlari var kostebek gibi https://t.co/4mEYYagl9F" #@terskaplumbaa
tweet_11 = "Neden? #biryudumkitap #kitap https://t.co/SZ5hEE9joU" #@biryudumkitapp
tweet_12 = "14 Aralik Cuma gunu 18.30'da, Zaman Degismeli sergimizde eserleriyle yer alan sanatci toplulugu Raqs Media Collec… https://t.co/jU0SMdFwwC" #@PeraMuzesi
tweet_13 = "Bu videoda izleyeceginiz insanlarin yasi 30'un altinda (ben haric :) fakat toplam takipci sayilari 15 milyonun uzer… https://t.co/WYXLhGZH0Q" #@BarisOzcan
tweet_14 = "@pianopiano01 basin sagolsun : ( \n\numarim simdiki uzuntunun yerine dostunla gecirdigin yillarin kiymetini gordugunde hafif bir mutluluk alir" #@mortifera
tweet_15 = "SOLOTuRK KKTC GoSTERISI 15 KASIM 2018: @YouTube araciligiyla https://t.co/DohPwVD2vg" #@soloturk
tweet_16 = "Tarihte ikinci kez, insan yapimi bir nesne, yildizlar arasindaki bosluga ulasti https://t.co/0505jB1Jqp" #@bilim_man
tweet_17 = "@cokkorkuyom offff agladim ben deee" #@iremsak
tweet_18 = "@waldeinn Kardesim biz almiyoruz da, ciddiye alacak hirt cok memlekette. Sorun da bu zaten." #@AtillaTasNet
tweet_19 = "RT @pyolum: Amazon verileri otomatik isaretleyen bir servis baslatmis? https://t.co/GNywYLqCTP" #@LaleAkarun
tweet_20 = "Gutenberg Firefox'ta asiri yavas. Bildigin makine otuyor. Uzunca yazi yazmiyorum sirf bu yuzden." #@yazilimci_adam

# a method that takes in a sentence and list of all words
# and returns the data in a form the can be fed to tensorflow

def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))

# we can start to predict the results for each of the 4 sentences
print(categories[np.argmax(model.predict([get_tf_record(tweet_1)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_2)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_3)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_4)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_5)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_6)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_7)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_8)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_9)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_10)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_11)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_12)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_13)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_14)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_15)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_16)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_17)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_18)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_19)]))])
print(categories[np.argmax(model.predict([get_tf_record(tweet_20)]))])
