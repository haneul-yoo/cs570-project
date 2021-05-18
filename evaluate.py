import os
import csv
import random
import time
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tqdm import tqdm

from transformers import XLMTokenizer, XLMWithLMHeadModel
from transformers import BertTokenizer, BertModel
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import ElectraTokenizer, ElectraModel
from tokenization_kobert import KoBertTokenizer
from tokenization_hanbert import HanBertTokenizer # download tokenization_hanbert.py

random_state = 42
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)

PRETRAINED_MAP = {
	'XLM': 'xlm-mlm-100-1280',
	'mBERT': 'bert-base-multilingual-cased',
	'XLM-RoBERTa': 'xlm-roberta-base',
	'KoBERT': 'monologg/kobert'
	'KoELECTRA': 'monologg/koelectra-base-v3-distriminator',
	'HanBERT': 'HanBERT-54kN-torch'
}

models = {
	'mBERT': BertModel.from_pretrained(PRETRAINED_MAP['mBERt'])
	'KoBERT': BertModel.from_pretrained(PRETRAINED_MAP['KoBERT']),
	'KoELECTRA': ElectraModel.from_pretrained(PRETRAINED_MAP['KoELECTRA']),
	'HanBERT': BertModel.from_pretrained(PRETRAINED_MAP['HanBERT'])
}

tokenizers = {
	'mBERT': BertTokenizer.from_pretrained(PRETRAINED_MAP['mBERT'], do_lower_case=False)
	'KoBERT': KoBertTokenizer.from_pretrained(PRETRAINED_MAP['KoBERT']),
	'KoELECTRA': ElectraTokenizer.from_pretrained(PRETRAINED_MAP['KoELECTRA']),
	'HanBERT': HanBertTokenizer.from_pretrained(PRETRAINED_MAP['HanBERT'])
}

def load_data():
	datas = {
		'train': pd.read_csv('./KorNLUDatasets/KorSTS/sts-train.tsv', sep='\t', encoding='UTF-8', skiprows=1, nrows=500),
		'dev': pd.read_csv('./KorNLUDatasets/KorSTS/sts-dev.tsv', sep='\t', encoding='UTF-8', skiprows=1),
		'test': pd.read_csv('./KorNLUDatasets/KorSTS/sts-test.tsv', sep='\t', encoding='UTF-8', skiprows=1)	
	}
	attention_masks = {
		'train': [],
		'dev': [],
		'test': []
	}
	tokenizer = tokenizers['KoBERT']
	MAX_LEN = 128

	for split in data.keys():
		datas[split] = ['[CLS]' + str(s) + '[SEP]' for s in datas[split]]
		datas[split] = [tokenizer.tokenize(s) for s in datas[split]]
		datas[split] = [tokenizer.convert_tokens_to_ids(x) for x in datas[split]]
		datas[split] = pad_sequences(datas[split], maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
		for seq in datas[split]:
			attention_masks.append([float(i > 0) for i in seq])
	return datas, attention_masks

def tokenize(data):


def accuracy(pred, true):
	pred = np.argmax(pred, axis=1).flatten()
	true = true.flatten()
	return np.sum(pred == true) / len(pred)

def format_time(elapsed):
	return str(datetime.timedelta(seconds=int(round(elapsed))))

def train():
	model = models['KoBERT']
	model.train()

	epochs = 20
	lr = 1e-5
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	total_loss = 0.0

	for epoch in range(epochs):
		print()
		print('====== Epoch {} / {} ======'.format(epoch + 1, epochs))
		print('Training...')

		t0

def test():


def main():


if __name__ == '__main__':
	main()