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
from tqdm import tqdm
ALPHABET = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
class MarvinOrganizerUtils:
    """Utilities class for the Marvin organizer"""
    pdfParser = ryltyPdfParser.ryltyNativePdfParser()
    def buildTextEmbedding(self,txt):
        def countDigramms(txt, k):
            """Return the number of occurences of k in txt."""
            # Count digramms 
            count = 0
            if len(k)==1:
                regEx= re.compile(" "+k+" ")
            else:
                regEx= re.compile(k)
            count =len(regEx.findall(txt))
            return count
        
        # build a feature vector based on digram statistics
        #1-build digramm list (single letter symbols are counted as well)
        digrammList = [d for d in ALPHABET] + [d1+d2 for d1 in ALPHABET for d2 in ALPHABET]
        # Remove special characters from text 
        txt = txt.replace('\n', ' ').lower()
        txt = unidecode(txt) # remove accents and special characters
        digrammCounts = [countDigramms(txt, k) for k in digrammList]
        #  digrammCounts = list(Parallel(n_jobs=-1)(delayed(countDigramms)(txt, k) for k in digrammList))
        digrammCountsArr = np.array(digrammCounts)
        embeddings = digrammCountsArr/np.sum(digrammCountsArr)
        return embeddings
    
    def buildFileEmbedding(self, filename):
        txt = ""
        emb = []
        try:
            if filename[-4:] == '.pdf':
                txt = self.readPdf(filename)
            elif filename[-4:] == '.csv':
                txt = self.readCSV(filename)
            elif filename[-4:] == '.xls' or filename[-5:] == '.xlsx':
                txt = self.readXLS(filename)
        except Exception as inst:
            emb = []
            print("Could not read : ", filename)
            print(type(inst))
            print(inst.args)
            print(inst)
            return emb
        emb = self.buildTextEmbedding(txt)
        return emb
    
    def readPdf(self, filename):
        self.pdfParser.loadPdf(filename)
        txt = ''.join(self.pdfParser.pageText)
        return txt
    
    def readCSV(self, filename):
        txt = ''
        with open(filename, "r", encoding="utf8", errors='ignore') as f:
            txt = f.read()
        return txt

    def readXLS (self, filename):
        df = pd.read_excel(filename)
        txt = df.to_string(index=False)
        return txt

class MarvinOrganizerData(Dataset):
    def __init__(self):
		# self.fileList = os.listdir(path)
		# self.fileData = {f:None for f in self.fileList}
        self.utils = MarvinOrganizerUtils()
        self.data = []
        self.label2id = {}
        self.id2label = {}

    def preloadFolder(self, path, classNum, label):
        """Load the data for one class from a given folder

        Args:
            path (str): path to the folder to load
            classNum (int): id of the class the data belongs to 
            label (str): the name of the class in plain text
        """
        jsonFile = path[:-1]+".json"
        folderData = []
        if os.path.exists(jsonFile):
            with open(jsonFile, "r") as f:
                folderData = json.load(f)
                # Force the class number and label in accordance to what is given in the function call
                for d in folderData:
                    d["classNum"] = classNum
                    d["lbl"] = label
        else:
            fileList = os.listdir(path)
            # for f in tqdm(fileList, total=len(fileList)):
            #     emb = utils.buildFileEmbedding(path+f)
            #     folderData.append({"vct": list(emb), "classNum": classNum, "lbl": label})
            embList = list((Parallel(n_jobs=-1)(delayed(utils.buildFileEmbedding)(path + f) for f in fileList)))
            folderData = [{"vct": list(v), "classNum": classNum, "lbl": label} for v in embList if (len(v) > 0) and (not np.isnan(v).any())]
            with open(jsonFile,"w") as f:
                json.dump(folderData, f)
        
        self.data = self.data + folderData
        self.label2id[label] = classNum
        self.id2label[classNum] = label

    def loadAll(self, root):
        """Load an entire directory tree at ones and infers the class name from the directory names.

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

class MarvinOrganizer(nn.Module):
    # Batch size when learning
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
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, nLabels)
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
    unitTest = False
    version = '2'
    modelPath = "./Model/"
    saveModelName = 'marvinOrganizer_'+version
    loadModelName = 'marvinOrganizer_'+version
    utils = MarvinOrganizerUtils()
    data = MarvinOrganizerData()
    if unitTest is True:
        rootPath = "./Data/Christopher Liggio/"
        data.preloadFolder(rootPath+"BMI Publisher/", classNum=0, label="BMI_Publisher")
        data.preloadFolder(rootPath+"BMI Writer/", classNum=1, label="BMI_Writer")
    else:
        rootPath = "/Fast/TrainData/RYLTY/Downloads/Organizer/"
        data.loadBySource(rootPath)
    # Load the  model if it exists
    marvin = MarvinOrganizer(nLabels=data.numClasses)
    if os.path.exists(modelPath+loadModelName+'.pth'):
        print("Loading: ", modelPath+loadModelName)
        marvin.load(modelPath, loadModelName)
    else:
        marvin.addLabelDescription(data)
    
    marvin.trainLoop(data,modelPath, saveModelName, epoch=2000) # 7000 Total
    marvin.save(modelPath, saveModelName)
    x = torch.rand(1,1260)
    pred = marvin.predict(x)
    print(pred)
