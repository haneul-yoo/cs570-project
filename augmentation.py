import os
import csv
import random
import nltk
# import konlpy
import MeCab
nltk.download('punkt')
from googletrans import Translator
from nltk.tokenize import word_tokenize

def readFiles():
	files = {
		'mnli': './KorNLUDatasets/KorNLI/multinli.train.ko.tsv',
		'snli': './KorNLUDatasets/KorNLI/snli_1.0_train.ko.tsv',
		'sts': './KorNLUDatasets/KorSTS/sts-train.tsv'
	}

	datas = {
		'mnli': [],
		'snli': [],
		'sts': []
	}
	for name, path in files.items():
		if name == 'sts':
			with open(path, 'r', encoding='utf-8') as f:
				reader = csv.DictReader(f, delimiter='\t')
				for line in reader:
					datas[name].append({
						'sentence1': line['sentence1'],
						'sentence2': line['sentence2'],
						'label': line['score']
					})
		else:
			with open(path, 'r', encoding='utf-8') as f:
				reader = csv.DictReader(f, delimiter='\t')
				for line in reader:
					datas[name].append({
						'sentence1': line['sentence1'],
						'sentence2': line['sentence2'],
						'label': line['gold_label']
					})
	return datas

def writeFiles(datas, augmentation_type):
	'''
	@ augmentation_type: 'EDA' or 'BT'
	'''
	dir_path = './augmentedDatasets/'
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	if not os.path.exists(dir_path + 'BT'):
		os.makedirs(dir_path + 'BT')
	if not os.path.exists(dir_path + 'EDA'):
		os.makedirs(dir_path + 'EDA')

	files = {
		'mnli': dir_path + augmentation_type + '/mnli_train_' + augmentation_type + '.tsv',
		'snli': dir_path + augmentation_type + '/snli_train_' + augmentation_type + '.tsv',
		'sts': dir_path + augmentation_type + '/sts_train_' + augmentation_type + '.tsv'
	}

	for name, path in files.items():
		with open(path, 'w', encoding='utf-8', newline='') as f:
			writer = csv.DictWriter(f, datas[name][0].keys(), delimiter='\t')
			writer.writeheader()
			writer.writerows(datas[name])


def backTranslation(datas, transfer=False):
	'''
	* pricing
		** google translate: 500,000 chars/month
		** naver papago: 10,000/day
	google translate api를 쓰도록 구현함
	하지만 무료 범위를 넘어가서 data를 줄이든, MT model 만들어서 직접 번역하든 해야 할 듯
	'''
	bt_datas = {
		'mnli': [],
		'snli': [],
		'sts': []
	}
	translator = Translator()
	for name, data in datas.items():
		for d in data:
			if transfer:
				bt_datas[name].append({
					'sentence1': translator.translate(translator.translate(translator.translate(translator.translate(d['sentence1'], src='ko', dest='ja').text, src='ja', dest='en').text, src='en', dest='ja').text, src='ja', dest='ko').text,
					'sentence2': translator.translate(translator.translate(translator.translate(translator.translate(d['sentence2'], src='ko', dest='ja').text, src='ja', dest='en').text, src='en', dest='ja').text, src='ja', dest='ko').text,
					'label': d['label']
				})
			else:
				bt_datas[name].append({
					'sentence1': translator.translate(translator.translate(d['sentence1'], src='ko', dest='en').text, src='en', dest='ko').text,
					'sentence2': translator.translate(translator.translate(d['sentence2'], src='ko', dest='en').text, src='en', dest='ko').text,
					'label': d['label']
				})

	return bt_datas

def EDA(datas):
	'''
	*한국어의 특징
		** 조사가 있어서 문장 내 단어 순서가 바뀌더라도 원 문장 의미를 유지한다
		** 의미가 단어 단위가 아닌 형태소 단위로 나뉜다
	따라서 random_swap이나 random_deletion은 어절 단위로(띄어쓰기 기준)
	synonym_replacement나 random_insertion은 형태소 단위로 한다
	'''
	def synonym_replacement(sent):
		return eda_sent

	def random_swap(sent):
		eda_sent = word_tokenize(sent)[:-1]
		if len(eda_sent) < 2:
			return sent
		[a, b] = random.sample(range(len(eda_sent)), 2)
		eda_sent[a], eda_sent[b] = eda_sent[b], eda_sent[a]
		eda_sent.append(sent[-1])
		return ' '.join(eda_sent)

	def random_insertion(sent):
		return eda_sent

	def random_deletion(sent):
		eda_sent = word_tokenize(sent)[:-1]
		if len(eda_sent) < 2:
			return sent
		eda_sent.pop(random.randrange(len(eda_sent)))
		eda_sent.append(sent[-1])
		return ' '.join(eda_sent)

	eda_datas = {
		'mnli': [],
		'snli': [],
		'sts': []
	}
	# eda_operations = [synonym_replacement, random_swap, random_insertion, random_deletion]
	eda_operations = [random_swap, random_deletion]
	for name, data in datas.items():
		for d in data[:500]:
			eda_datas[name].append({
				'sentence1': random.choice(eda_operations)(d['sentence1']),
				'sentence2': random.choice(eda_operations)(d['sentence2']),
				'label': d['label']
			})
	return eda_datas

def main():
	random_state = 42
	random.seed(random_state)
	# backTranslation(readFiles())
	writeFiles(EDA(readFiles()), 'EDA')

if __name__ == '__main__':
	main()