import os
import json
import pandas as pd
import numpy as np
# Paralellize some jobs where possible... after all I got 32 CPUs I should use them...
from joblib import Parallel, delayed
import re
from unidecode import unidecode
from gensim.models.fasttext import FastText
from tqdm import tqdm
from scipy.special import softmax

import gemsimUtils

CST_MODEL_VERSION = "V1"
CST_VCT_TYPE = "FastTxtAvg"

class MarvinOrganizerUtils:
    """Utilities class for the Marvin organizer"""
    # We skip pdf for now
    # pdfParser = ryltyPdfParser.ryltyNativePdfParser()

    def __init__(self):
        self.fastTxtModel = FastText.load("./Model/marvinFastTxt.gs")
        # This is a dirty hack against some XL errors 
        # Set it to some value (eg. False) to deactivate
        self.fixXLErr14 = None

    def normalizeString(self, rawStr:str):
        """Remove accents, non alphnumeric characters, normalize white spaces and to lower case"""
        strN = unidecode(rawStr)
        strN = re.sub(r"\W", " ", strN).lower().strip()
        strN = " ".join(strN.split())
        return strN

    def vctToString(self, vct:list):
        """convert a vector to a nicely formated string with a well defined resolution

        Args:
            vct (list): list of array of float values
        Returns:
            str: a string representing the vector
        """
        depth = 3
        vct = np.round(np.array(vct) * 10**depth) / 10** depth
        vctStr = "["+", ".join([str(v) for v in vct])+"]"
        return vctStr
    
    def buildTextEmbedding(self,txt):
        if len(txt.strip()) == 0:
            return None
        wLstClean = txt.split()
        embMtx  = []
        for i, w in enumerate(wLstClean):
            emb = self.fastTxtModel.wv[w] 
            embMtx.append(emb)
        embMtx = np.array(embMtx)
        embV = np.sum(embMtx, axis=0)
        embV = embV/np.linalg.norm(embV)
        return embV
        # if len(txt.strip()) == 0:
        #     return None
        # wLst = txt.split()
        # embMtx  = []
        # for w in wLst:
        #     emb = self.fastTxtModel.wv[w]
        #     embMtx.append(emb)

        # embMtx = np.array(embMtx)
       
        # MM = np.matmul(embMtx.T, embMtx)
        # eigVal, eigVct = np.linalg.eig(MM)
        # return np.real(eigVct[:,0])
    
    def rebuildDFHeader(self, df:pd.DataFrame):
        """Check if the data frame contains a header before the genuine data and drops de corresponding lines

        Args:
            df (pd.DataFrame): dataframe to fix
        Returns:
            (pd.DataFrame): the same data frame with incomplete lines at the beginnning taken out.
        """
        cols = list(df.columns)
        cols = [str(c) for c in cols]
        fixme = False
        for c in cols:
            if "Unnamed" in c:
                fixme = True
                break
        if fixme is True:
            firstRow = 0
            # Count the number of of items on each line
            valCount = [np.sum(np.logical_not(pd.isna(np.array(df.iloc[r])))) for r in range(df.shape[0])]
            # We assume that the longest list is the most likely to be the header (especially if we find it close to the top of the file)
            firstRow = np.argmax(valCount)
            if (firstRow > 0) and (firstRow < len(valCount) - firstRow):
                if (valCount[firstRow+1] - valCount[firstRow-1] <= 2):
                    firstRow = 0
                else:
                    realHeaders = df.iloc[firstRow]
                    df = df.iloc[firstRow+1:]
                    df.columns = realHeaders
        return df

    def buildFileEmbedding(self, filename):
        txt = ""
        df = None
        emb = []
        ext = filename.split('.')[-1].lower()
        try:
            # if ext == 'pdf':
            #     txt = self.readPdf(filename)
            if ext == 'csv':
                df = self.readCSV(filename)
            elif ext == 'xls' or ext == 'xlsx':
                df = self.readXLS(filename)
        except Exception as inst:
            emb = []
            print("Could not read : ", filename)
            print(type(inst))
            print(inst)
            return emb
        if df is not None:
            cols = list(df.columns)
            cols = [str(c) for c in cols]
            txt = " ".join(cols)
            emb = self.buildTextEmbedding(txt)
        return emb, txt, filename
    
    def readPdf(self, filename):
        self.pdfParser.loadPdf(filename)
        txt = ''.join(self.pdfParser.pageText)
        return txt
    
    def readCSV(self, filename):
        sepLst=[',',';','\t']
        df = None
        for s in sepLst:
            if df is None:
                try:
                    df = pd.read_csv(filename, sep=s,engine='python', encoding="utf_8", low_memory=False)
                except:
                    try:
                        df = pd.read_csv(filename, sep=s,engine='c', encoding="utf_8", low_memory=False)
                    except:
                        try:
                            df = pd.read_csv(filename, sep=s,engine='python', encoding='latin_1', low_memory=False)
                        except:
                            try:
                                df = pd.read_csv(filename, sep=s,engine='c', encoding='latin_1', low_memory=False)
                            except:
                                df = None
        return df

    def readXLS (self, filename):
        # This is a potential patch for stylesheet related errors taken from:
        # https://stackoverflow.com/questions/50236928/openpyxl-valueerror-max-value-is-14-when-using-load-workbook/71526058#71526058
        # IMPORTANT, you must do this before importing openpyxl
        from unittest import mock
        # Set max font family value to 100
        if self.fixXLErr14 is None:
            self.fixXLErr14 = mock.patch('openpyxl.styles.fonts.Font.family.max', new=100)
        self.fixXLErr14.start()
        df = None
        try:
            df = pd.read_excel(filename, engine='calamine')
        except:
            try:
                df = pd.read_excel(filename, engine='openpyxl')
            except:
                try:
                    df = pd.read_excel(filename, engine='xlrd')
                except:
                    try:
                        df =  pd.read_excel(filename, engine='pyxlsb')
                    except:
                        df = None
        self.fixXLErr14.stop()
        return df

class MarvinOrganizerData():

    def __init__(self):
        self.utils = MarvinOrganizerUtils()
        self.data = []
        self.txt = []
        self.label2id = {}
        self.id2label = {}

    def preloadFolder(self, path, classNum, label):
        """Load the data for one class from a given folder

        Args:
            path (str): path to the folder to load
            classNum (int): id of the class the data belongs to 
            label (str): the name of the class in plain text
        """
        jsonFile = path[:-1] + "_" + str(CST_MODEL_VERSION) + "_" + CST_VCT_TYPE+".json"
        txtFile = path[:-1] + "_" + str(CST_MODEL_VERSION)+ "_" + CST_VCT_TYPE+".txt"
        folderData = []
        folderTxt = []
        if os.path.exists(jsonFile):
            with open(jsonFile, "r") as f:
                folderData = json.load(f)
                # Force the class number and label in accordance to what is given in the function call
                for d in folderData:
                    d["classNum"] = classNum
                    d["lbl"] = label
            # with open(txtFile, "r") as f:
            #     folderTxt = f.readlines()
        else:
            fileList = os.listdir(path)
            embList = list((Parallel(n_jobs=-1)(delayed(utils.buildFileEmbedding)(path + f) for f in fileList)))
            if len(embList) > 0:
                folderData = [{ "file": v[2],
                                "txt":v[1],
                                "vct": [float(i) for i in v[0]],
                                "classNum": classNum, "lbl": label} for v in embList if (len(v) > 0) and (len(v[0]) > 0)]

        if len(folderData) > 0:
            self.vecSize = len(folderData[0]["vct"])
            self.data = self.data + folderData
            # self.txt = self.txt + folderTxt
            self.label2id[label] = classNum
            self.id2label[classNum] = label
            with open(jsonFile,"w") as f:
                json.dump(folderData, f)

    def loadAll(self, root):
        """Load an entire directory tree and infers the class name from the directory names.

            This assumes a fixed directory tree: type/Source
            In this one the class is a composite of the file type and source: 

        Args:
            root (str): root directory to start exploring from (must end with "/").
        """
        classIdx = 0
        for d1 in os.listdir(root):
            for d2 in os.listdir(root+d1):
                self.preloadFolder(root+d1+"/"+d2, classNum=classIdx, label=d1+"-"+d2)
                classIdx += 1
   
    def loadByType(self, root):
        """Load an entire directory tree at ones and sort the data by type.

            This assumes a fixed directory tree: type/Source

        Args:
            root (str): root directory to start exploring from (must end with "/").
        """
        classIdx = 0
        for d1 in os.listdir(root):
            for d2 in os.listdir(root+d1):
                self.preloadFolder(root+d1+"/"+d2, classNum=classIdx, label=d1)
            classIdx+=1
        self.numClasses = classIdx

    def loadBySource(self, root):
        """Load an entire directory tree at ones and sort the data by source.

            This assumes a fixed directory tree: type/Source

        Args:
            root (str): root directory to start exploring from (must end with "/").
        """
        classIdx = 0
        sourceIdx = {}
        for d1 in os.listdir(root):
            if os.path.isdir(root+d1):
                for d2 in os.listdir(root+d1):
                    if os.path.isdir(root+d1+"/"+d2):
                        if d2 in list(sourceIdx.keys()):
                            self.preloadFolder(root+d1+"/"+d2+"/", classNum=sourceIdx[d2], label=d2)
                        else:
                            self.preloadFolder(root+d1+"/"+d2+"/", classNum=classIdx, label=d2)
                            sourceIdx[d2] = classIdx
                            classIdx += 1
        self.numClasses = classIdx

    def len(self):
        return len(self.data)
    

class MarvinOrganizer():
    minCossim = 1 # Only acccept perfectMatches 
    def __init__(self):
        self.utils = MarvinOrganizerUtils()
        pass

    def loadModel(self, modelFile:str):
        with open(modelFile, 'r') as f:
            self.formatModels = json.load(f)

    def fromDataset(self, dataset:MarvinOrganizerData):
        self.id2label= dataset.id2label
        self.label2id = dataset.label2id
        self.sourceVectors = {k:[] for k in self.label2id.keys()}
        self.sourceProba = {k:0 for k in self.label2id.keys()}
        self.vectorSources = {}
        self.vectorSourcesProba = {}
        self.vectorProba = {}
        print("Building model...")
        for s in tqdm(dataset.data, total=dataset.len()):
            v = s['vct']
            lbl = s['lbl']
            k = utils.vctToString(v)
            self.sourceProba[lbl] += 1
            self.sourceVectors[lbl] += [v]
            if k in self.vectorSources.keys():
                if lbl not in self.vectorSources[k]:
                    self.vectorSources[k] += [lbl]
                    self.vectorSourcesProba[k][lbl] = 1
                else: 
                    self.vectorSourcesProba[k][lbl] += 1
                self.vectorProba[k] += 1
            else:
                self.vectorSources[k] = [lbl]
                self.vectorSourcesProba[k] = {lbl:1}
                self.vectorProba[k] = 1
 
        # Compute the prior probability of each class based on the frequency in the test dataset
        srcProba = [self.sourceProba[k] for k in self.sourceProba.keys()]
        srcProba = np.exp(srcProba/np.sum(srcProba)) # Data is normalized to a unit vector to keep it numericaly behaved
        srcProba = srcProba/np.sum(srcProba)
        for i, k in enumerate(self.sourceProba.keys()):
            self.sourceProba[k] = srcProba[i]
        # Compute the prior probability of each class
        vctCnt = [self.vectorProba[k] for k in self.vectorProba.keys()]
        vctCnt = np.exp(vctCnt / np.sum(vctCnt))# Data is normalized to a unit vector to keep it numericaly behaved
        vctCnt = vctCnt / np.sum(vctCnt)
        for (i,k) in enumerate(self.vectorProba.keys()):
            self.vectorProba[k] = vctCnt[i]
        # Compute the conditionnal probablility P(Lbl|Vct)
        for srcK in self.vectorSources.keys():
            lblsCnts = [self.vectorSourcesProba[srcK][lbl] for lbl in self.vectorSourcesProba[srcK].keys()]
            lblsCnts = np.exp(lblsCnts / np.sum(lblsCnts))# Data is normalized to a unit vector to keep it numericaly behaved
            lblsCnts = lblsCnts / np.sum(lblsCnts)
            for i, k in enumerate(self.vectorSourcesProba[srcK].keys()):
                self.vectorSourcesProba[srcK][k] = lblsCnts[i]
    
    def getSources(self, fileName:str):
        """Retrieve the royalty sources based on the format signature. 

        Args:
            fileName (str): path to the file to process

        Returns:
            (list): list of labels (royalty sources)
            (list): posterior probability of each hypothesis
        """
        emb, _, _ = self.utils.buildFileEmbedding(fileName)
        sourcesList = []
        sourcesProba = [] 
        if len(emb) == 0 or np.isnan(np.any(emb)):
            return sourcesList, sourcesProba

        embStr = utils.vctToString(emb)
        if embStr in self.vectorSources.keys():
            sources = self.vectorSources[embStr]
            # Compute the posterior of the match based on the computed statistics on the training data set
            # We drop the source prior for now since it looks very disbalanced
            # sourcesProba = [(self.vectorSourcesProba[embStr][k] * self.sourceProba[k] ) /self.vectorProba[embStr] for k in sources]
            sourcesProba = [self.vectorSourcesProba[embStr][k] for k in sources]
            sourcesProba = np.exp(sourcesProba)
            sourcesProba = sourcesProba/np.sum(sourcesProba)
        return sources, sourcesProba
 
    def toJson(self, jsonFile:str):
        """Save the current configuration to a json file

        Args:
            jsonFile (str): path to the json file
        """
        model = {"sourceVectors": self.sourceVectors,
                 "sourceProbas": self.sourceProba,
                 "vectorProba": self.vectorProba,
                 "vectorSources": self.vectorSources,
                 "vectorSourcesProba": self.vectorSourcesProba} 
        with open(jsonFile,'w') as f:
            json.dump(model, f)
          

    def fromJson(self, jsonFile:str):
        """Save the current configuration to a json file

        Args:
            jsonFile (str): path to the json file
        """
        with open(jsonFile,'r') as f:
            model = json.load(f)
        self.sourceVectors= model["sourceVectors"]
        self.sourceProba = model["sourceProbas"]
        self.vectorProba = model["vectorProba"]
        self.vectorSources = model["vectorSources"]
        self.vectorSourcesProba = model["vectorSourcesProba"]

    def addLabelDescription(self, dataset:MarvinOrganizerData):
        """Add label/id information to the model from the training dataset

        Args:
            dataset (MarvinOrganizerData): dataset that will be used for training
        """
        self.label2id = dataset.label2id
        self.id2label = dataset.id2label

if __name__ == "__main__":
    # Select task
    marvinMatchTest = True
    trainFastTxt = False
    version = 'v1.1'
    modelPath = "./Model/"
    saveModelName = 'marvinOrganizer_'+"_FastTxt"+version
    loadModelName = 'marvinOrganizer_'+"_FastTxt"+version
    signatureModelName = 'marvinOrganizerSignatures_'+CST_VCT_TYPE+version+".json"
    utils = MarvinOrganizerUtils()
    data = MarvinOrganizerData()
    if trainFastTxt is True:
        rootPath = "/Fast/TrainData/RYLTY/Downloads/Organizer/"
        jsPath = "/Fast/TrainData/RYLTY/Downloads/Organizer/Statement"
        data.loadBySource(rootPath)
        fastTxtModel = FastText.load("./Model/marvinFastTxt.gs")
        corpus = gemsimUtils.loadCorpusTxt(jsPath)
        # build the vocabulary
        fastTxtModel.build_vocab(corpus_iterable=corpus)
        fastTxtModel.build_vocab(corpus_iterable=corpus)

        # train the model
        fastTxtModel.train(
            corpus_iterable=corpus, epochs=5000,
            total_examples=fastTxtModel.corpus_count, total_words=fastTxtModel.corpus_total_words,
        )
        fastTxtModel.save("./Model/marvinFastTxt.gs")
        vct = fastTxtModel.wv["kitty kat"]
        print(vct)
    if marvinMatchTest is True:
        marvin = MarvinOrganizer()
        if os.path.exists(modelPath+signatureModelName): # load the model
            marvin.fromJson(modelPath+signatureModelName)
        else: # Build the dataset 
            rootPath = "/Fast/TrainData/RYLTY/Downloads/Organizer/"
            data.loadBySource(rootPath)
            marvin.fromDataset(data)
            marvin.toJson(modelPath+signatureModelName)

        # rootPath = "./Data/Christopher Liggio/"
        rootPath = "/Fast/TrainData/RYLTY/Downloads/Organizer/Statement/"
        print("+=====================================+")
        print(" BMI Publisher: ")
        for f in os.listdir(rootPath+"BMI Publisher/"):
            source, sourceScore = marvin.getSources(rootPath+"BMI Publisher/"+f)
            print(f, "->", source,": ", sourceScore)
            
        print("+=====================================+")
        print(" BMI Writer: ")

        for f in os.listdir(rootPath+"BMI Writer/"):
            source, sourceScore = marvin.getSources(rootPath+"BMI Writer/"+f)
            print(f, "->", source,": ", sourceScore)
        

