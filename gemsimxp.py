# Build a Fasttext feature vector based on the Marvin Organizer's vocabulary.
from pprint import pprint as print
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath

from unidecode import unidecode
import glob
import json
import re

jsPath = "/Fast/TrainData/RYLTY/Downloads/Organizer/Statement"
def normalizeString(rawStr:str):
	"""Remove accents, non alphnumeric characters, normalize white spaces and to lower case"""
	strN = unidecode(rawStr)
	strN = re.sub(r"\W", " ", strN).lower().strip()
	return strN

def loadCorpus(path):
	txt = []
	for jsonFile in glob.glob(path+'/*.json'):
		with open(jsonFile, "r") as js:
			jsData = json.load(js)
		if len(jsData) > 0:
			jsTxt = [j["txt"] for j in jsData]
			txt = txt + jsTxt
	txt = [normalizeString(t) for t in txt]
	return txt 

model = FastText(vector_size=100)
corpus = loadCorpus(jsPath)
# build the vocabulary
model.build_vocab(corpus_iterable=corpus)
model.build_vocab(corpus_iterable=corpus)

# train the model
model.train(
    corpus_iterable=corpus, epochs=5000,
    total_examples=model.corpus_count, total_words=model.corpus_total_words,
)
model.save("./Model/marvinFastTxt.gs")
vct = model.wv["kitty kat"]
print(vct)
