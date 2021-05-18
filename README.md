## Project

### Data

| Source | Data  | Split | # Examples | Avg. # of words |
| ------ | ----- | ----- | ---------- | --------------- |
| KorNLI | SNLI  | Train | 550153     | 10.8            |
| KorNLI | MNLI  | Train | 392703     | 10.8            |
| KorNLI | XNLI  | Dev   | 2490       | 13.0            |
| KorNLI | XNLI  | Test  | 5010       | 13.1            |
| KorSTS | STS-B | Train | 5479       | 7.5             |
| KorSTS | STS-B | Dev   | 1500       | 8.7             |
| KorSTS | STS-B | Test  | 1379       | 7.6             |

Train data는 Machine Translation의 결과
Dev, Test data는 MT를 Human translator가 검수

Data Augmentation에서 다룰 데이터는 Train Data only

Original Paper에서는 500, 1k, 2k, 3k, all 일 때의 성능 변화를 확인 

### Data Augmentation

#### Back Translation

* translation API
  (우선 Google Translate를 쓰도록 구현했지만 무료 범위를 넘어섬)
  * Google Translate: 500,000 chars/month
  * Naver Papago: 10,000/day

* 한국어는 Low Resource Language
  * 한국어→영어 번역보다 한국어→일본어→영어 로의 번역이 더 잘 된다는 통념
    * 한국어가 일본어에 비해 low resource language
    * 한국어-일본어 간의 문법이 유사하여 Machine Translation에 유리

#### Easy Data Augmentation

* 한국어의 특징을 고려한 EDA implementation
  * 조사가 있어서 문장 내 단어 순서가 바뀌더라도 원 문장 의미를 유지한다
    → random swap, random deletion은 어절 단위(띄어쓰기 기준)으로 한다
  * 의미가 단어 단위가 아닌 형태소 단위로 나뉜다
    → synonym replacement, random insertion은 형태소 단위로 한다
    * expected problem) 한국어에 마땅한 synonym dictionary가 없다
      오히려 단어를 영어로 바꿔서 동의어를 찾고 이 동의어를 번역하는 게 추천될 정도..
    * 그나마 국립국어원에서 공개한 <우리말 샘> 사전이 있지만, 실제 사용하기는 힘든 수준
    * 

### Pretrained Model

Original paper에서는 BERT, RoBERTa, XLNet으로 비교했다
MLRG 발표 때 ELECTRA를 같이 비교했어야 한다는 feedback이 있었다

|Name|

MultilingualBERT, XLM, XLM-R, KoBERT(SKT), KorBERT(ETRI), KR-BERT(SNU), HanBERT(twoblockai), KoELECTRA