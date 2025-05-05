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

CST_MODEL_VERSION = "V4"
CST_VCT_TYPE = "FastTxtHash"
#@TODO Remove files built with terylty template:
# statementDate,title,workId,incomeSource,sourceGrossIncome,royaltyAmount,incomeType,distributionType,featuredArtist,country,perfCount

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
        vct = np.round(np.array(vct) * 10**depth) / 10**depth
        vctStr = "["+", ".join([str(v) for v in vct])+"]"
        return vctStr
    
    def vctToHash(self, vct:list):
        """convert a vector to a hash code

        Args:
            vct (list): list of array of float values
        Returns:
            int: hash code representing the vector
        """
        depth = 10**5
        vctA = np.array(vct)
        # Reduce the floating point precision so that we can deal with the precisiojn loss when loading/saving the data.
        hashKey = hash(tuple(np.round(vctA*depth)/depth)) 
        return hashKey
    
    def buildTextEmbedding(self,txt):
        """Build a vector representation of a text of arbitrary lenght.

            The resulting vector is the unit lenght average of the fastText representation of each word in the text

        Args:
            txt (str): input text

        Returns:
            array: vector representation of the text
        """
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

           @WARNING: EXPERIMENTAL CODE NOT KNOWN TO WORK VERY RELIABLY

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
        col2idx = {}
        idx2col = {}
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
            col2idx = {c:i for i,c in enumerate(cols)}
            idx2col = {i:c for i,c in enumerate(cols)}
            emb = self.buildTextEmbedding(txt)
        return emb, col2idx, idx2col, filename
    
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
                    df = pd.read_csv(filename, sep=s,engine='python', encoding="utf_8", nrows=10)
                except:
                    try:
                        df = pd.read_csv(filename, sep=s,engine='c', encoding="utf_8", nrows=10, low_memory=False)
                    except:
                        try:
                            df = pd.read_csv(filename, sep=s,engine='python', encoding='latin_1', nrows=10, low_memory=False)
                        except:
                            try:
                                df = pd.read_csv(filename, sep=s,engine='c', encoding='latin_1', nrows=10, low_memory=False)
                            except:
                                df = None
            if df is not None and df.shape[1] == 1:
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
            df = pd.read_excel(filename, nrows=10, engine='calamine')
        except:
            try:
                df = pd.read_excel(filename, nrows=10, engine='openpyxl')
            except:
                try:
                    df = pd.read_excel(filename, nrows=10, engine='xlrd')
                except:
                    try:
                        df =  pd.read_excel(filename, nrows=10, engine='pyxlsb')
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
        else:
            fileList = os.listdir(path)
            embList = list((Parallel(n_jobs=-1)(delayed(utils.buildFileEmbedding)(path + f) for f in fileList)))
            if len(embList) > 0:
                folderData = [{ "file": v[3],
                                "col2idx":v[1],
                                "idx2col":v[2],
                                "vct": [float(i) for i in v[0]],
                                "classNum": classNum, "lbl": label} for v in embList if (len(v) > 0) and (len(v[0]) > 0)]

        if len(folderData) > 0:
            self.vecSize = len(folderData[0]["vct"])
            self.data = self.data + folderData
            # self.txt = self.txt + folderTxt
            self.label2id[label] = classNum
            self.id2label[classNum] = label
            with open(jsonFile,"w") as jf:
               json.dump(folderData, jf)

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
        """Basic constructor, nothing to see here.
        """
        self.utils = MarvinOrganizerUtils()
        pass

    def updateFromFile(self, file:str, label:str):
        """Add a single file to the model.

            If the format is not know, a new category is created 
            If the format is known, the corresponding priors are updated.


        Args:
            file (str): string or path to the file
            label (str): label of the file (statement source)
        """
        emb, col2id, id2col, _ = self.utils.buildFileEmbedding(file)
        embStr = str(utils.vctToHash(emb))
        if label in self.sourceCount.keys():
            self.sourceCount[label] += 1
            self.sourceVectors[label] += [emb]
        else:
            self.sourceCount[label] = 1
            self.sourceVectors[label] = [emb]
        if embStr in self.vectorSources.keys():
            if label not in self.vectorSources[embStr]:
                self.vectorSources[embStr] += [label]
                self.vectorSourcesCount[embStr][label] = 1
            else: 
                self.vectorSourcesCount[embStr][label] += 1
            self.vectorCount[embStr] += 1
        else:
            self.vectorSources[embStr] = [label]
            self.vectorSourcesCount[embStr] = {label:1}
            self.vectorCount[embStr] = 1
            self.col2idx[embStr] = col2id
            self.idx2col[embStr] = id2col
        self.__countsToProba__()

    def fromDataset(self, dataset:MarvinOrganizerData):
        self.id2label= dataset.id2label
        self.label2id = dataset.label2id
        self.sourceVectors = {k:[] for k in self.label2id.keys()}
        self.sourceCount = {k:0 for k in self.label2id.keys()}
        # self.sourceProba = {k:0 for k in self.label2id.keys()}
        self.col2idx = {}
        self.idx2col = {} 
        self.vectorSources = {}
        self.vectorSourcesCount = {}
        self.vectorSourcesProba = {}
        self.vectorCount = {}
        self.vectorProba = {}
        self.col2idx = {}
        print("Building model...")
        for s in tqdm(dataset.data, total=dataset.len()):
            v = s['vct']
            lbl = s['lbl']
            embHash = str(utils.vctToHash(v))
            self.col2idx[embHash] = s['col2idx']
            self.idx2col[embHash]= s['idx2col']
            self.sourceCount[lbl] += 1
            self.sourceVectors[lbl] += [v]
            if embHash in self.vectorSources.keys():
                if lbl not in self.vectorSources[embHash]:
                    self.vectorSources[embHash] += [lbl]
                    self.vectorSourcesCount[embHash][lbl] = 1
                else: 
                    self.vectorSourcesCount[embHash][lbl] += 1
                self.vectorCount[embHash] += 1
            else:
                self.vectorSources[embHash] = [lbl]
                self.vectorSourcesCount[embHash] = {lbl:1}
                self.vectorCount[embHash] = 1
        self.__countsToProba__()

    def __countsToProba__(self):
        """Utility function, convert vector and source counts to probability distributions.

            Uses Softmax for that.
            Also updates self.lbl2id and self.id2lbl to match class names (rylty sources) and the label number
        """
        # Compute the prior probability of each class based on the frequency in the test dataset
        srcProba = [self.sourceCount[k] for k in self.sourceCount.keys()]
        srcProba = np.exp(srcProba/np.sum(srcProba)) # Data is normalized to a unit vector to keep it numericaly behaved
        srcProba = srcProba/np.sum(srcProba)
        self.sourceProba = {k:p for k, p in zip(self.sourceCount.keys(), srcProba)}
        # Compute the prior probability of each embedding vector
        vctProba = [self.vectorCount[k] for k in self.vectorCount.keys()]
        vctProba = np.exp(vctProba / np.sum(vctProba))# Data is normalized to a unit vector to keep it numericaly behaved
        vctProba = vctProba / np.sum(vctProba)
        self.vectorProba = {k:p for k, p in zip(self.vectorCount.keys(), vctProba)}
        # Compute the conditionnal probablility P(Vct|Lbl)
        self.vectorSourcesProba = {}
        for srcK in self.vectorSources.keys():
            lblsProba = [self.vectorSourcesCount[srcK][lbl] for lbl in self.vectorSourcesCount[srcK].keys()]
            lblsProba = np.exp(lblsProba / np.sum(lblsProba))# Data is normalized to a unit vector to keep it numericaly behaved
            lblsProba = lblsProba / np.sum(lblsProba)
            self.vectorSourcesProba[srcK] = {k:p for k, p in zip(self.vectorSourcesCount[srcK].keys(), lblsProba)}
        for i,k in enumerate(self.sourceProba.keys()):
            self.label2id[k] = i
            self.id2label[i] = k

    def getSources(self, fileName:str):
        """Retrieve the royalty sources based on the format signature. 

        Args:
            fileName (str): path to the file to process
        Returns:
            (list): list of labels (royalty sources)
            (list): posterior probability of each hypothesis
        """
        emb, _, _, _ = self.utils.buildFileEmbedding(fileName)
        sourcesList = []
        sourcesProba = [] 
        if len(emb) == 0 or np.isnan(np.any(emb)):
            return sourcesList, sourcesProba

        embStr = str(utils.vctToHash(emb))
        if embStr in self.vectorSources.keys():
            sourcesList = self.vectorSources[embStr]
            # Compute the posterior of the match based on the computed statistics on the training data set
            # We drop the source prior for now since it looks very disbalanced
            # sourcesProba = [(self.vectorSourcesProba[embStr][k] * self.sourceProba[k] ) /self.vectorProba[embStr] for k in sources]
            sourcesProba = [self.vectorSourcesProba[embStr][k] for k in sourcesList]
            sourcesProba = np.exp(sourcesProba)
            sourcesProba = sourcesProba/np.sum(sourcesProba)
        return sourcesList, sourcesProba
 
    def toJson(self, jsonFile:str):
        """Save the current configuration to a json file

        Args:
            jsonFile (str): path to the json file
        """
        model = {"sourceVectors": self.sourceVectors,
                "sourceCount": self.sourceCount,
                "vectorCount": self.vectorCount,
                "vectorSources": self.vectorSources,
                "vectorSourcesCount": self.vectorSourcesCount,
                "col2idx":self.col2idx,
                "idx2col":self.idx2col
                #  "label2id": self.label2id,
                #  "id2label": self.id2label
                } 
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
        self.sourceCount = model["sourceCount"]
        self.vectorCount = model["vectorCount"]
        self.vectorSources = model["vectorSources"]
        self.vectorSourcesCount = model["vectorSourcesCount"]
        self.col2idx = model['col2idx']
        self.idx2col = model['idx2col']
        
        # These will be set by self.__countsToProba__()

        self.label2id = {} #model["label2id"]
        self.id2label = {} #model["id2label"]
        self.__countsToProba__()

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
    version = 'v4.0'
    modelPath = "./Model/"
    saveModelName = 'marvinOrganizer_'+"_FastTxt"+version
    loadModelName = 'marvinOrganizer_'+"_FastTxt"+version
    signatureModelName = 'marvinOrganizerSignatures_'+CST_VCT_TYPE+version+".json"
    utils = MarvinOrganizerUtils()
    data = MarvinOrganizerData()
    if trainFastTxt is True:
        rootPath = "/Fast/TrainData/RYLTY/Organizer/"
        jsPath = "/Fast/TrainData/RYLTY/Organizer/Statement"
        print("Loading data")
        data.loadBySource(rootPath)
        fastTxtModel = FastText.load("./Model/marvinFastTxt.gs")
        corpus = gemsimUtils.loadCorpusTxt(jsPath)
        # build the vocabular
        print("Building Fasttext corpus")
        fastTxtModel.build_vocab(corpus_iterable=corpus)
        fastTxtModel.build_vocab(corpus_iterable=corpus)

        # train the model
        print("Training...")
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

        rootPath = "/Fast/TrainData/RYLTY/Organizer/Statement/"
        print("+=====================================+")
        print(" Believe Digital: ")
        for f in os.listdir(rootPath+"Believe Digital/"):
            source, sourceScore = marvin.getSources(rootPath+"Believe Digital/"+f)
            # Note updating for a know source would only update the priors, 
            # is it really worth it while we work with exact matches only ? 
            # marvin.updateFromFile(rootPath+"Believe Digital/"+f, label= "Believe Digital")
            print(f, "->", source,": ", sourceScore)
            if len(source) == 0:
                marvin.updateFromFile(rootPath+"Believe Digital/"+f, label= "Goofy Records!!")
                print(" Added label: Goofy Records!!")
                source, sourceScore = marvin.getSources(rootPath+"Believe Digital/"+f)
                print(f, "->", source,": ", sourceScore)
