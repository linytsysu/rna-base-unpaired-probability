import math, random
from const import START, STOP

import numpy as np
from collections import defaultdict, OrderedDict
from pprint import pprint

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
#from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay

class Network(Layer):
    def __init__(self,
                 sequence_vocabulary, bracket_vocabulary, mixture_vocabulary,
                 dmodel=128,
                 layers=8,
                 dropout=0.15,
                 ):
        super(Network, self).__init__()
        self.sequence_vocabulary = sequence_vocabulary
        self.bracket_vocabulary = bracket_vocabulary
        self.mixture_vocabulary = mixture_vocabulary
        self.dropout_rate = dropout
        self.model_size = dmodel
        self.layers = layers

    def lstm_subnet(self, emb):
        emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu")
        for i in range(self.layers):
            emb = paddle.fluid.layers.fc(emb, size=self.model_size*2)
            fwd, cell  = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size*2, use_peepholes=True, is_reverse=False)
            back, cell = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size*2, use_peepholes=True, is_reverse=True)
            emb = paddle.fluid.layers.concat(input=[fwd, back], axis=1)
            emb = paddle.fluid.layers.dropout(emb, self.dropout_rate)
            emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu")
        return emb

    def conv_subnet(self, emb):
        conv_3 = paddle.fluid.layers.sequence_conv(emb, num_filters=self.model_size, filter_size=3, act="relu")
        conv_4 = paddle.fluid.layers.sequence_conv(input=emb, num_filters=self.model_size, filter_size=4, act="relu")
        conv_5 = paddle.fluid.layers.sequence_conv(input=emb, num_filters=self.model_size, filter_size=5, act="relu")
        emb = paddle.fluid.layers.concat(input=[conv_3, conv_4, conv_5], axis=1)
        emb = paddle.fluid.layers.dropout(emb, self.dropout_rate)
        emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu")
        return emb

    def conv_pooling_subnet(self, emb):
        conv3 = paddle.fluid.nets.sequence_conv_pool(emb, self.model_size, 3, act="tanh")
        conv4 = paddle.fluid.nets.sequence_conv_pool(emb, self.model_size, 4, act="tanh")
        conv5 = paddle.fluid.nets.sequence_conv_pool(emb, self.model_size, 5, act="tanh")
        emb = paddle.fluid.layers.concat(input=[conv3, conv4, conv5], axis=1)
        emb = paddle.fluid.layers.dropout(emb, self.dropout_rate)
        emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu")
        return emb

    def forward(self, seq, dot, mix):
        emb_seq = paddle.fluid.embedding(seq, size=(self.sequence_vocabulary.size, self.model_size), is_sparse=True)
        emb_dot = paddle.fluid.embedding(dot, size=(self.bracket_vocabulary.size, self.model_size), is_sparse=True)
        emb_mix = paddle.fluid.embedding(mix, size=(self.mixture_vocabulary.size, self.model_size), is_sparse=True)
        conv_seq = self.conv_subnet(emb_seq)
        conv_dot = self.conv_subnet(emb_dot)
        conv_mix = self.conv_subnet(emb_mix)

        lstm_seq = self.lstm_subnet(emb_seq)
        lstm_dot = self.lstm_subnet(emb_dot)
        lstm_mix = self.lstm_subnet(emb_mix)

        emb1 = paddle.fluid.layers.concat(input=[conv_seq,conv_dot,conv_mix], axis=1)
        emb1 = paddle.fluid.layers.fc(emb1, size=self.model_size, act="relu")

        emb2 = paddle.fluid.layers.concat(input=[lstm_seq,lstm_dot,lstm_mix], axis=1)
        emb2 = paddle.fluid.layers.fc(emb2, size=self.model_size, act="relu")

        emb = paddle.fluid.layers.concat(input=[emb1, emb2], axis=1)

        ff_out = paddle.fluid.layers.fc(emb, size=2, act="relu")
        soft_out = paddle.fluid.layers.softmax(ff_out, axis=1)
        return soft_out[:,0]
