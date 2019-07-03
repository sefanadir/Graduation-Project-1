#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import data_helpers
import gc
from input_helpers import InputHelper
from text_cnn import TextCNN
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
# Parameters
# ==================================================

tf.flags.DEFINE_string("word2vec", "GoogleNews-vectors-negative300.singles.gz", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("word2vec_format", "textgz", "Word2vec pretrained file format. textgz: gzipped text | bin: binary format (default: textgz)")
tf.flags.DEFINE_boolean("word2vec_trainable", False, "Allow modification of w2v embedding weights (True/False)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '2,3,4')")
tf.flags.DEFINE_string("filter_h_pad", 5, "Pre-padding for each filter (default: 5)")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_integer("max_document_words", 600, "Max length (left to right max words to consider) in every doc, else pad 0 (default: 100)")
tf.flags.DEFINE_string("training_files", None, "Comma-separated list of training files (each file is tab separated format) (default: None)")
tf.flags.DEFINE_integer("hidden_units", 50, "Number of hidden units in softmax regression layer (default:50)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.training_files==None:
    print ("Input Files List is empty. use --training_files argument.")
    exit()
 
training_paths=FLAGS.training_files.split(",")

multi_train_size = len(training_paths)
max_document_length = FLAGS.max_document_words

inpH = InputHelper()
train_set, dev_set, vocab_processor,sum_no_of_batches = inpH.getDataSets(training_paths, max_document_length, FLAGS.filter_h_pad, 10, FLAGS.batch_size)
inpH.loadW2V(FLAGS.word2vec, FLAGS.word2vec_format)
# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=max_document_length,
            num_classes=2,
            multi_size = multi_train_size,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            hidden_units=FLAGS.hidden_units,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            retrain_emb=FLAGS.word2vec_trainable)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        print("initialized cnn object")
    grad_set=[]
    tr_op_set=[]
    for i2 in xrange(multi_train_size):
        #optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars=optimizer.compute_gradients(cnn.loss[i2])
        tr_op_set.append(optimizer.apply_gradients(grads_and_vars, global_step=global_step))
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.merge_summary(grad_summaries)
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

    # Write vocabulary
    vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.initialize_all_variables())
    
    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)

    if FLAGS.word2vec:
        # initial matrix with random uniform
        initW = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        # load any vectors from the word2vec
        print("init initW cnn.W in FLAG")
        for w in vocab_processor.vocabulary_._mapping:      
            arr=[] 
            s = re.sub('[^0-9a-zA-Z]+', '', w)
            if w in inpH.pre_emb:
                arr=inpH.pre_emb[w]
            elif w.lower() in inpH.pre_emb:
                arr=inpH.pre_emb[w.lower()] 
            elif s in inpH.pre_emb:
                arr=inpH.pre_emb[s]
            elif s.isdigit():
                arr=inpH.pre_emb["1"]
            if len(arr)>0:
                idx = vocab_processor.vocabulary_.get(w)
                initW[idx]=np.asarray(arr).astype(np.float32)
        print("assigning initW to cnn. len="+str(len(initW)))
        inpH.deletePreEmb()
        gc.collect()
        sess.run(cnn.W.assign(initW))

    def train_step(x_batch, y_batch, typeIdx):
        feed_dict = {
                             cnn.input_x: x_batch,
                             cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
        }
        for i in xrange(multi_train_size):
            if i==typeIdx:
                feed_dict[cnn.input_y[i]] = y_batch
            else:
                feed_dict[cnn.input_y[i]] = np.zeros((len(x_batch),2))
         
        _, step, loss, accuracy, pred = sess.run([tr_op_set[typeIdx], global_step, cnn.loss[typeIdx], cnn.accuracy[typeIdx], cnn.predictions[typeIdx]],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("TRAIN {}: type {}, step {}, loss {:g}, acc {:g}".format(time_str, typeIdx, step, loss, accuracy))
        print (np.argmax(y_batch, 1), pred)
        #train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch, y_batch, typeIdx, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
                         cnn.input_x: x_batch,
                         cnn.dropout_keep_prob: 1.0,
        }
        for i in xrange(multi_train_size):
            if i==typeIdx:
                    feed_dict[cnn.input_y[i]] = y_batch
            else:
                    feed_dict[cnn.input_y[i]] = np.zeros((len(x_batch),2))            

        step, loss, accuracy, pred = sess.run([global_step, cnn.loss[typeIdx], cnn.accuracy[typeIdx], cnn.predictions[typeIdx]],  feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("DEV {}: type {}, step {}, loss {:g}, acc {:g}".format(time_str, typeIdx, step, loss, accuracy))
        print (np.argmax(y_batch, 1), pred)
        #if writer:
        #    writer.add_summary(summaries, step) 
        return accuracy

    # Generate batches
    batches=[]
    for i in xrange(multi_train_size):
        batches.append(data_helpers.batch_iter(
                list(zip(train_set[i][0], train_set[i][1])), FLAGS.batch_size, FLAGS.num_epochs))

    ptr=0
    max_validation_acc=0.0
    for nn in xrange(sum_no_of_batches*FLAGS.num_epochs):
        idx=round(np.random.uniform(low=0, high=multi_train_size))
        if idx<0 or idx>multi_train_size-1:
            continue
        typeIdx = int(idx)
        print (typeIdx)
        batch = batches[typeIdx].next()
        if len(batch)<1:
            continue
        x_batch, y_batch = zip(*batch)
        if len(y_batch)<1:
            continue
        train_step(x_batch, y_batch,typeIdx)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc=0.0
        if current_step % FLAGS.evaluate_every == 0:
            for dtypeIdx in xrange(multi_train_size):

            	print("\nEvaluation:")
            	dev_batches = data_helpers.batch_iter(list(zip(dev_set[dtypeIdx][0],dev_set[dtypeIdx][1])), 2*FLAGS.batch_size, 1)
            	for db in dev_batches:
                    if len(db)<1:
                        continue
                    x_dev_b,y_dev_b = zip(*db)
                    if len(y_dev_b)<1:
                        continue
                    acc = dev_step(x_dev_b, y_dev_b, dtypeIdx)
                    sum_acc = sum_acc + acc
            	print("")
        if current_step % FLAGS.checkpoint_every == 0:
            if sum_acc >= max_validation_acc:
                max_validation_acc = sum_acc
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
                print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc, checkpoint_prefix))
