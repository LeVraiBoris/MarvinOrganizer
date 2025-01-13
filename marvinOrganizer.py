import os
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
        digrammCounts = list(Parallel(n_jobs=-1)(delayed(countDigramms)(txt, k) for k in digrammList))
        digrammCountsArr = np.array(digrammCounts)
        embeddings = digrammCountsArr/np.sum(digrammCountsArr)
        return embeddings
    
    def buildFileEmbedding(self, filename):
        txt = ""
        if filename[-4:] == '.pdf':
            txt = self.readPdf(filename)
        elif filename[-4:] == '.csv':
            txt = self.readCSV(filename)
        elif filename[-4:] == '.xls' or filename[-5:] == '.xlsx':
            txt = self.readXLS(filename)
        emb = self.buildTextEmbedding(txt)
        return emb
    
    def readPdf(self, filename):
        self.pdfParser.loadPdf(filename)
        txt = ''.join(self.pdfParser.pageText)
        return txt
    
    def readCSV(self, filename):
        df = pd.read_csv(filename)
        txt = df.to_string(index=False)
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

    def preloadFolder(self, path, label):
        fileList = os.listdir(path)

        for f in tqdm(fileList, total=len(fileList)):
            emb = utils.buildFileEmbedding(path+f)
            self.data.append({"vct": emb, "lbl": label})

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        idx = idx % self.__len__()
        vct = torch.tensor(self.data[idx]["vct"])

        lbl = self.data[idx]["lbl"]
        return vct, lbl

class MarvinOrganizer(nn.Module):
    # Batch size when learning
    batchSize = 1
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
        return classNumber


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
            if i % 100 == 0:
                loss, current = loss.item(), i * self.batchSize + len(embedding)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
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

    def trainLoop(self, dataset, epoch=10):
        """Train the model using the provided dataset 

        Args:
            dataset (MarvinOrganizerData): train/test data
            epoch (int): number of epoch to run the training. defualts to 10
        """
        generator = torch.Generator().manual_seed(42)
        trainData, testData = torch.utils.data.random_split(dataset, [0.8, 0.2])
        trainDataLoader = DataLoader(trainData, batch_size=self.batchSize, shuffle=True)
        testDataLoader = DataLoader(testData, batch_size=self.batchSize, shuffle=True)
        for e in range(epoch):
            print(f"Epoch {e+1}\n-------------------------------")
            self.fit(trainDataLoader)
            self.test(testDataLoader)
        print("Done !")
# class ryltyEncoderSorter(ryltySongSorter):
#     """Perform song/alternate income source sorting using a sparse autoencoder model"""

#     def __init__(self, modelFile=None):
#         super(ryltyEncoderSorter).__init__()
#         print('Loading model ....', end=' ')
#         self.model = ryltySparseAutoEncoder()
#         self.model.load_state_dict(torch.load(modelFile))
#         # setting device on GPU if available, else CPU
#         self.torchDevice = 'cuda' if torch.cuda.is_available() else 'cpu'
#         # Send the network to GPU/CPUself.encode(x)
#         self.model.to(self.torchDevice)
#         # Set the model in evaluation mode and disable autograd36*36
#         self.model.eval()
#         torch.no_grad()
#         print('Done.')

#     def sortSongs(self, songList):
#         self.rawSongList = songList
#         # Encode items in the song list
#         self.uniqueSongNames, self.uniqueSongIndexes = self.utils.getUniqueSongNames(self.rawSongList)
#         embeddings = self.LLM.encode(self.uniqueSongNames)
#         encodings = [self.model.encode(emb) for emb in embeddings]
#         output = [self.model.decode(c) for c in encodings]

#         # See what is going on
#         for song, encoding in zip(self.uniqueSongNames, encodings):
#             print(song, ": ", encoding)

if __name__ == "__main__":
    version = '1.0'
    modelPath = "./Model/"
    saveModelName = 'marvinOrganizer_'+version+".pth"
    loadModelName = 'marvinOrganizer_'+version+".pth"
    rootPath = "./Data/Christopher Liggio/"
    utils = MarvinOrganizerUtils()
    marvin = MarvinOrganizer()
    if os.path.exists(modelPath+loadModelName):
        marvin.load_state_dict(torch.load(modelPath+loadModelName, weights_only=True))
        marvin.toDevice()

    data = MarvinOrganizerData()
    data.preloadFolder(rootPath+"BMI Publisher/", label=0)
    data.preloadFolder(rootPath+"BMI Writer/", label=1)
    marvin.trainLoop(data,epoch=10)
    torch.save(marvin.state_dict(), modelPath+saveModelName)
    x = torch.rand(1,1260).to()
    pred = marvin.predict(x)
    print(pred)
    # emb = utils.buildFileEmbedding('./test.pdf')
    # print('Pdf: ')
    # for i in emb:
    #      print(i, end=", ")
    
    # emb = utils.buildFileEmbedding('./test.xlsx')
    # print('\n Xls: ')
    # for i in emb:
    #      print(i, end=", ")
    # emb = utils.buildFileEmbedding('./test.csv')
    # print('\n Csv: ')
    # for i in emb:
    #      print(i, end=", ")