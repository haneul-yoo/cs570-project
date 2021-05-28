import os
import glob
import csv
import json
import random
import xmltodict
from natsort import natsorted
import logging
logging.basicConfig(level=logging.INFO)

def write_csv(data):
	with open('./synonym_dataset.csv', 'w', encoding='UTF-8', newline='') as f:
		writer = csv.DictWriter(f, data[0].keys())
		writer.writeheader()
		writer.writerows(data)

def build_synonym_dataset():
	path = './우리말샘_xml/*'
	xmls = [file for file in glob.glob(path) if file.endswith('.xml')]
	xmls = natsorted(xmls)
	synonym = []
	logging.info('List all xml files')

	for xml in xmls:
		with open(xml, 'r', encoding='UTF-8') as f:
			data = json.loads(json.dumps(xmltodict.parse(f.read())['channel']['item']))
			for item in data:
				if 'relation_info' in item['senseInfo']:
					if isinstance(item['senseInfo']['relation_info'], list):
						word = item['wordInfo']['word']
						replaced = []
						# relation = relation_info['type']
						wtype = item['senseInfo']['pos'] if 'pos' in item['senseInfo'] else None
						for relation_info in item['senseInfo']['relation_info']:
							replaced.append(relation_info['word'])
						synonym.append({'word': word,
										'replaced': replaced,
										# 'relation': relation,
										'type': wtype})
					elif item['senseInfo']['relation_info']['type'] == '비슷한말':
						synonym.append({'word': item['wordInfo']['word'],
										'replaced': [item['senseInfo']['relation_info']['word']],
										# 'relation': item['senseInfo']['relation_info']['type'],
										'type': item['senseInfo']['pos'] if 'pos' in item['senseInfo'] else None})
		logging.info(xml + ' Done')
	write_csv(synonym)

def main():
	build_synonym_dataset()

if __name__ == '__main__':
	main()
