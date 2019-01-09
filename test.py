# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from transformer import Transformer
import tensorflow as tf


with open('zh-en.en', 'r', encoding='utf8') as f:
    data = f.readlines()

from tqdm import tqdm

en_list = []
for line in tqdm(data[:100000]):
    for c in ['\n', '\t', '\xa0', '&nbsp', '\xad', '�', '\u200b', '\u3000', '\x9d']:
        line = line.replace(c, '')
    en_list.append(line)

for line in en_list[100:110]: print(line)

"""- 英文分词

一般我们按照空格分词即可，但是英文中出现了数字和一些符号不能用空格进行分词，因此需要加入空格到这些数字和符号中去。
"""

def en_pre(en_list):

    symbol_list = [',', ':', '.', '?', '!', ')', '(', '-', '“', '”', '’', '‘',
                   '/', '"', '\'', '\\', '–', ';', '[', ']', '—', '…', '@', '#',
                   '$', '&', '*', '_', '=']
    num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '%']

    for i in range(len(en_list)):
        line = en_list[i]
        line = line.lower()
        for c in symbol_list+num_list:
            line = line.replace(c, ' ' + c + ' ')
        en_list[i] = line
    return en_list

en_list = en_pre(en_list)

for line in en_list[100:110]: print(line)

"""- 构造英文字典"""

en_data = []
for line in tqdm(en_list):
    line = line.split(' ')
    line = [word for word in line if word not in ['', ' ']]
    en_data.append(line)

for line in en_data[100:110]:
    print(line)

en_vocab = {'<PAD>':0}
index = 0
for line in tqdm(en_data):
    for word in line:
        if word not in en_vocab:
            index += 1
            en_vocab[word] =  index


"""- zh数据获取"""

with open('zh-en.zh', 'r', encoding='utf8') as f:
    data = f.readlines()

from tqdm import tqdm

symbol_list = ['\xa0', '\n', ' ', '\t', '\u200b', '\u3000', '\xad']
zh_list = []
for line in tqdm(data[:100000]):
    for c in symbol_list:
        line = line.replace(c, '')
    zh_list.append(line)

for line in zh_list[:10]: print(line)

"""- 构造中文字典"""

zh_vocab = {'<PAD>':0, '<GO>':1, '<EOS>':2}
index = 2
for line in tqdm(zh_list):
    for char in line:
        if char not in zh_vocab:
            index += 1
            zh_vocab[char] = index

print(len(zh_vocab))

print(len(zh_list))
print(len(en_data))

"""## 1.2 构建数据生成器"""

encoder_inputs = [[en_vocab[word] for word in line] for line in tqdm(en_data)]
decoder_inputs = [[zh_vocab['<GO>']] + [zh_vocab[word] for word in line] for line in tqdm(zh_list)]
decoder_targets = [[zh_vocab[word] for word in line] + [zh_vocab['<EOS>']] for line in tqdm(zh_list)]

print(decoder_inputs[:4])
print(decoder_targets[:4])

import numpy as np

def get_batch(encoder_inputs, decoder_inputs, decoder_targets, batch_size=4):
    batch_num = len(encoder_inputs) // batch_size
    for k in range(batch_num):
        begin = k * batch_size
        end = begin + batch_size
        en_input_batch = encoder_inputs[begin:end]
        de_input_batch = decoder_inputs[begin:end]
        de_target_batch = decoder_targets[begin:end]
        max_en_len = max([len(line) for line in en_input_batch])
        max_de_len = max([len(line) for line in de_input_batch])
        en_input_batch = np.array([line + [0] * (max_en_len-len(line)) for line in en_input_batch])
        de_input_batch = np.array([line + [0] * (max_de_len-len(line)) for line in de_input_batch])
        de_target_batch = np.array([line + [0] * (max_de_len-len(line)) for line in de_target_batch])
        yield en_input_batch, de_input_batch, de_target_batch

batch = get_batch(encoder_inputs, decoder_inputs, decoder_targets, batch_size=4)
next(batch)



"""# 构建模型"""

#from transformer import Transformer

def create_hparams():
    params = tf.contrib.training.HParams(
        num_heads = 8,
        num_blocks = 6,
        # vocab
        input_vocab_size = 50,
        label_vocab_size = 50,
        # embedding size
        max_length = 100,
        hidden_units = 512,
        dropout_rate = 0.1,
        lr = 0.0001,
        is_training = True)
    return params


arg = create_hparams()
arg.input_vocab_size = len(en_vocab)
arg.label_vocab_size = len(zh_vocab)
arg.is_training = False
arg.dropout_rate = 0.


g = Transformer(arg)

saver =tf.train.Saver()

de_zh_vocab = {v : k for k, v in zh_vocab.items()}

with tf.Session() as sess:
    saver.restore(sess, 'logs/model')
    for i in range(100):
        line = encoder_inputs[i*1000]
        x = np.array(line)
        x = x.reshape(1, -1)
        de_inp = [[zh_vocab['<GO>']]]
        while True:
            y = np.array(de_inp)
            preds = sess.run(g.preds, {g.x: x, g.de_inp: y})
            if preds[0][-1] == zh_vocab['<EOS>']:
                break
            de_inp[0].append(preds[0][-1])
        got = ''.join(de_zh_vocab[idx] for idx in de_inp[0][1:])
        print('英文原文：', ''.join(en_list[i*1000]))
        print('中文原文：', zh_list[i*1000])
        print('识别结果：', got)
