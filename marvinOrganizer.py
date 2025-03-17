import os
import json
import pandas as pd
import numpy as np
# Paralellize some jobs where possible... after all I got 32 CPUs I should use them...
from joblib import Parallel, delayed
from pathlib import Path
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import torch.utils
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import ryltyPdfParser
from unidecode import unidecode
from gensim.models.fasttext import FastText
from tqdm import tqdm
from scipy.special import softmax

import gemsimUtils

ALPHABET = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9',' ']

CST_MODEL_VERSION = "V1"
CST_VCT_TYPE = "FastTxtEigV"
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

    def buildTextEmbedding(self,txt):
        if len(txt.strip()) == 0:
            return None
        wLst = txt.split()
        embMtx  = []
        for w in wLst:
            emb = self.fastTxtModel.wv[w]
            embMtx.append(emb)

        embMtx = np.array(embMtx)
       
        MM = np.matmul(embMtx.T, embMtx)
        eigVal, eigVct = np.linalg.eig(MM)
        return np.real(eigVct[:,0])
    
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
                    df = pd.read_csv(filename, sep=s,engine='python', encoding="utf_8")
                except:
                    try:
                        df = pd.read_csv(filename, sep=s,engine='c', encoding="utf_8")
                    except:
                        try:
                            df = pd.read_csv(filename, sep=s,engine='python', encoding='latin_1')
                        except:
                            try:
                                df = pd.read_csv(filename, sep=s,engine='c', encoding='latin_1')
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
        # import openpyxl
        # openpyxl.open('my-bugged-worksheet.xlsx') # this works now!

        df = None
        try:
            df = pd.read_excel(filename)
        except:
            df = pd.read_excel(filename, engine='openpyxl')
        self.fixXLErr14.stop()
        return df

class MarvinOrganizerData(Dataset):

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
                                "classNum": classNum, "lbl": label} for v in embList if (len(v) > 0) and v[0] is not None]

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
            for d2 in os.listdir(root+d1):
                if os.path.isdir(root+d1+"/"+d2):
                    if d2 in list(sourceIdx.keys()):
                        self.preloadFolder(root+d1+"/"+d2+"/", classNum=sourceIdx[d2], label=d2)
                    else:
                        self.preloadFolder(root+d1+"/"+d2+"/", classNum=classIdx, label=d2)
                        sourceIdx[d2] = classIdx
                        classIdx += 1
        self.numClasses = classIdx

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        idx = idx % self.__len__()
        vct = torch.tensor(self.data[idx]["vct"])

        cls = self.data[idx]["classNum"] # Class Index
        return vct, cls
    
    def len(self):
        return len(self.data)
    

class MarvinOrganizer():

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
        print("Loading Building model ...")
        for s in tqdm(dataset.data, total=dataset.__len__()):
            v = s['vct']
            lbl = s['lbl']
            k = str(v)
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
            lblsCnts = np.exp(lblsCnts / np.sum(lblsCnts))
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
        if len(emb) == 0 or np.isnan(np.any(emb)):
            return (None, None)
        embStr = str(emb)
        if embStr in self.vectorSources.keys():
            sources = self.vectorSources[embStr]
            # Compute the posterior of the match based on the computed statistics on the training data set
            sourcesProba = [(self.vectorSourcesProba[embStr][k] * self.vectorProba[embStr]) / self.sourceProba[k] for k in sources]
            return sources, sourcesProba, [1 for s in sources]
        else:
            # Get the closest match
            bestScore = 0
            lbl = None
            for s in self.sourceVectors.keys():
                for v in self.sourceVectors[s]:
                    score = np.dot(emb/np.linalg.norm(emb),v/np.linalg.norm(v))
                    if score>bestScore:
                        bestScore = score
                        lbl = s
                        key = str(v)
            sourcesProba = self.vectorSourcesProba[key][lbl] * self.vectorProba[key] / self.sourceProba[lbl]
            return [lbl], [sourcesProba], [bestScore]

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


    batchSize = 16
    learningRate = 1e-3
    # Loss regularization
    regParam = 1e-3
    # the loss function
    criterion = nn.MSELoss()

    def __init__(self, inputSize=1260, nLabels=2):
        """Constructor

        @WARNING: Input and output layer size are hard coded wrt the embedding size in the current language model (768 for BERT)

        Args:
            inputSize (int): length of the input vector (text embedding). Should be 1296. Defaults to 1296
            nLabels (int): the number of classes to train the network upon. Defaults to 2
        """
        super(MarvinOrganizer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(inputSize, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(256,128),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(512, nLabels)
        )
        self.softMax = nn.Softmax(dim=1)
        self.childrenList = list(self.children())
        # Send model to GPU
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        self.toDevice()
        # Initialize the loss function and optimizer 
        self.lossFct = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learningRate)
        self.label2id = {}
        self.id2label = {}

    def addLabelDescription(self, dataset:MarvinOrganizerData):
        """Add label/id information to the model from the training dataset

        Args:
            dataset (MarvinOrganizerData): dataset that will be used for training
        """
        self.label2id = dataset.label2id
        self.id2label = dataset.id2label

    def toDevice(self):
        """Send the model to GPU if available"""
        self.to(self.device)
        self.to(torch.float32)

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.to(self.device)
        x = self.network(x)
        return x

    def predict(self, x):
        x = x.to(torch.float32)
        x = x.to(self.device)
        logits = self.network(x)
        predProbab = self.softMax(logits)
        classNumber = predProbab.argmax(1)
        score = predProbab[classNumber]
        classLabel = self.id2label[str(classNumber)]
        return classNumber, classLabel, score


    def fit(self, dataloader):
        """Train the model for one epoch

        Args:
            dataset (Dataset): torch.dataset instance, in principle built from a ryltySongData
            epoch (int): number of epoch to run training

        Returns:
            _type_: _description_
        """
        print('Training')
        
        size = len(dataloader.dataset)
        self.train()
        running_loss = 0.0
        counter = 0
        for i, (data, lbl) in tqdm(enumerate(dataloader), total=int(size/dataloader.batch_size)):
            counter += 1
            embedding = data
            embedding = embedding.to(self.device).to(torch.float32)
            lbl = lbl.to(self.device).to(torch.long)
            self.optimizer.zero_grad()
            outputs = self(embedding)
            loss = self.lossFct(outputs, lbl)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # if i % 100 == 0:
            #     loss, current = loss.item(), i * self.batchSize + len(embedding)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        epoch_loss = running_loss / counter
        print(f"Train Loss: {loss:.3f}")
        return epoch_loss

    def test(self, dataloader):
        """Test the model on the provided data for one epoch

        Args:
            dataset (Dataset): torch.dataset instance, in principle built from a ryltySongData
        Returns:
            _type_: _description_
        """
        print('Test')
        size = len(dataloader.dataset)
        self.eval()
        runningLoss, correct = 0, 0
        with torch.no_grad():
            for i, (data, lbl) in tqdm(enumerate(dataloader), total=int(size/dataloader.batch_size)):
                embedding = data.to(self.device).to(torch.float32)
                lbl = lbl.to(self.device).to(torch.long)
                self.optimizer.zero_grad()
                outputs = self(embedding)
                loss = self.lossFct(outputs, lbl)
                runningLoss += loss.item()
                correct += (outputs.argmax(1) == lbl).type(torch.float).sum().item()
        epochLoss = runningLoss / len(dataloader)
        epochCorrect = correct / size
        print(f"Test Error: \n Accuracy: {(100*epochCorrect):>0.1f}%, Avg loss: {epochLoss:>8f} \n")
        return epochLoss
    
    def trainLoop(self, dataset, modelPath, modelName, epoch=10):
        """Train the model using the provided dataset 

        Args:
            dataset (MarvinOrganizerData): train/test data
            modelPath (str): path to the folder where to autosave the model
            modelName (str): base name for the model
            epoch (int): number of epoch to run the training. Defaults to 10
        """
        # self.addLabelDescription(dataset)
        trainData, testData = torch.utils.data.random_split(dataset, [0.8, 0.2])
        trainDataLoader = DataLoader(trainData, batch_size=self.batchSize, shuffle=True)
        testDataLoader = DataLoader(testData, batch_size=self.batchSize, shuffle=True)
        bestLoss = 10000
        for e in range(epoch):
            print(f"Epoch {e+1}\n-------------------------------")
            self.fit(trainDataLoader)
            epochLoss = self.test(testDataLoader)
            if epochLoss < bestLoss:
                self.save(modelPath, modelName)
                bestLoss = epochLoss
        print("Done !")

    def save(self, path, modelName):
        """Save the model and the label to id information in a common place

        Args:
            path (str): path to the folder to save the model
            modelName (str): name of the model (basename -without extension- for the different files that will be created)
            
        """
        torch.save(self.state_dict(), modelPath+modelName+'.pth')
        with open(modelPath+modelName+'.json','w') as f:
            json.dump([self.label2id, self.id2label], f)

    def load(self, path, modelName):
        """load the model and the label to id information in a common place

        Args:
            path (str): path to the folder to save the model
            modelName (str): name of the model (basename -without extension- for the different files to load)
            
        """
        self.load_state_dict(torch.load(modelPath+modelName+'.pth', weights_only=True))
        # Note: we do not do any data type conversion since any error there means we do not load the correct model (see constructor)
        self.toDevice()

        with open(modelPath+modelName+'.json','r') as f:
            lblid  = json.load(f)
            self.label2id = lblid[0]
            self.id2label = lblid[1]

if __name__ == "__main__":
    # Select task
    marvinMatchTest = False
    trainFastTxt = True
    version = 'v1'
    modelPath = "./Model/"
    saveModelName = 'marvinOrganizer_'+"_FastTxt"+version
    loadModelName = 'marvinOrganizer_'+"_FastTxt"+version
    signatureModelName = 'marvinSigns_'+CST_VCT_TYPE+version+".json"
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

        rootPath = "./Data/Christopher Liggio/"
        print("+=====================================+")
        print(" BMI Publisher: ")
        for f in os.listdir(rootPath+"BMI_Publisher/"):
            src, sourceScore, matchScore = marvin.getSources(rootPath+"BMI_Publisher/"+f)
            print(f)
            for i,s in enumerate(src):
                print(s," :",sourceScore[i],", ", matchScore[i])
        print("+=====================================+")
        print(" BMI Writer: ")

        for f in os.listdir(rootPath+"BMI_Writer/"):
            src, sourceScore, matchScore = marvin.getSources(rootPath+"BMI_Writer/"+f)
            print(f)
            for i,s in enumerate(src):
                print(s," :",sourceScore[i],", ", matchScore[i])
        
        

