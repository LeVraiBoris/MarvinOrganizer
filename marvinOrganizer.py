import os
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from  ..ocrUtils import ryltyPdfParser

from tqdm import tqdm

class MarvinOrganizerUtils:
    """Utilities class for the Marvin organizer"""
    pdfPArser = ryltyPdfParser.ryltyNativePdfParser()
    def buildFileEmbedding(self, filename):
        self.pdfPArser = ryltyPdfParser.ryltyNativePdfParser()

    def readPdf(self, filename):
       ryltyPdfParser.loadPdf(filename)
       txt = ryltyPdfParser.getTextFromPage(1)
class MarvinOrganizerData(Dataset):
	def __init__(self, path):
		self.fileList = os.listdir(path)
		self.fileData = {f:None for f in self.fileList}
	def __len__(self):
		return len(self.fileList)
    
	def __getitem__(self,idx):
		idx = idx % self.__len__()
		if self.fileData[self.fileList[idx]] is None:
			self.fileData[self.fileList[idx]] = MarvinOrganizerUtils.buildFileEmbedding(self.fileList[idx])
		return self.fileData[self.fileList[idx]]
     
class ryltySparseAutoEncoder(nn.Module):
    # Batch size when learning
    batchSize = 32
    learningRate = 1e-3
    # Loss regularization
    regParam = 1e-3
    # the loss function
    criterion = nn.MSELoss()

    def __init__(self, sparse=True):
        """Constructor

        @WARNING: Input and output layer size are hard coded wrt the embedding size in the current language model (768 for BERT)

        Args:
            sparse (bool, optional): If true a sparse loss function is used when training. Defaults to True.
        """
        super(ryltySparseAutoEncoder, self).__init__()
        self.sparse = sparse
        # encoder
        self.enc1 = nn.Linear(in_features=768, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=16)

        # decoder
        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=768)

        self.childrenList = list(self.children())


        # Send model to GPU
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        self.to(self.device)

        # the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learningRate)

    def encode(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        return(x)

    def decode(self, x):
        # decoding
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        return x

    def forward(self, x):
        # encoding
        x = self.encode(x)
        # decoding
        x = self.decode(x)
        return x


    def sparse_loss(self, embedding):
        loss = 0
        values = embedding
        for i in range(len(self.childrenList)):
            values = F.relu((self.childrenList[i](values)))
            loss += torch.mean(torch.abs(values))
        return loss

    def fit(self, dataset, epoch):
        """Train the model on the provided data

        Args:
            dataset (Dataset): torch.dataset instance, in principle built from a ryltySongData
            epoch (int): number of epoch to run training

        Returns:
            _type_: _description_
        """
        print('Training')
        dataloader = DataLoader(dataset, batch_size=self.batchSize, shuffle=True)
        self.train()
        running_loss = 0.0
        counter = 0

        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            embedding = data
            embedding = embedding.to(self.device)
            embedding = embedding.view(embedding.size(0), -1)
            self.optimizer.zero_grad()
            outputs = self(embedding)
            mse_loss = self.criterion(outputs, embedding)
            if self.sparse is True:
                l1_loss = self.sparse_loss(embedding)
                # add the sparsity penalty
                loss = mse_loss + self.regParam * l1_loss
            else:
                loss = mse_loss
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / counter
        print(f"Train Loss: {loss:.3f}")
        return epoch_loss

    def test(self, dataset, epoch):
        """Train the model on the provided data

        Args:
            dataset (Dataset): torch.dataset instance, in principle built from a ryltySongData
            epoch (int): number of epoch to run training

        Returns:
            _type_: _description_
        """
        print('Training')
        dataloader = DataLoader(dataset, batch_size=self.batchSize, shuffle=True)
        self.eval()
        runningLoss = 0.0
        counter = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
                counter += 1
                embedding = data
                embedding = embedding.to(self.device)
                embedding = embedding.view(embedding.size(0), -1)
                self.optimizer.zero_grad()
                outputs = self(embedding)
                mseLoss = self.criterion(outputs, embedding)
                if self.sparse is True:
                    l1Loss = self.sparse_loss(embedding)
                    # add the sparsity penalty
                    loss = mseLoss + self.regParam * l1Loss
                else:
                    loss = mseLoss
                runningLoss += loss.item()

        epochLoss = runningLoss / counter
        print(f"Train Loss: {loss:.3f}")
        return epochLoss

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
#         # Set the model in evaluation mode and disable autograd
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
     utils = MarvinOrganizerUtils()
     utils.readPdf('./test.pdf')