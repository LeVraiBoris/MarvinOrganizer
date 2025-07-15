"""A Script to generate a test dataset based based on original->transformed file pairs. Organized by rylty source.
"""
import marvinOrganizer as marvin
import os
import numpy as np
import shutil
import sys
sourcePath = "/Fast/TrainData/RYLTY/Downloads/Translator"
testPath = "/Fast/TrainData/RYLTY/AppTestDataset"

organizer = marvin.MarvinOrganizer()
utils = marvin.MarvinOrganizerUtils()

version = marvin.CST_MODEL_VERSION
modelPath = "./Model/"
saveModelName = 'marvinOrganizer_'+"_FastTxt"+version
loadModelName = 'marvinOrganizer_'+"_FastTxt"+version
signatureModelName = 'marvinOrganizerSignatures_'+marvin.CST_VCT_TYPE+version+".json"
if os.path.exists(modelPath+signatureModelName): # load the model
    organizer.fromJson(modelPath+signatureModelName)
    print("Model Loaded..")
else: 
    print("Marvin model not found, exiting.")
    sys.exit()

if os.path.exists(testPath):
    shutil.rmtree(testPath)
os.mkdir(testPath)

# Iterate over files in directory

for dirPath, dList, fList in os.walk(sourcePath):
    for f in fList:
        if not f.startswith("transformed"):
            filePath = os.path.join(dirPath, f)
            source, scores = organizer.getSources(filePath)
            if len(source) > 0:
                sourceIdx = np.argmax(scores)
                if scores[sourceIdx] > marvin.CST_MIN_CONFIDENCE: # if the file source is known
                    trfFile = os.path.join(dirPath, "transformed_"+f[:-4]+'.csv')
                    if os.path.exists(trfFile): # if the corresponding transformed file exists
                        destDir = os.path.join(testPath, source[sourceIdx])
                        if not os.path.isdir(destDir): # if the destination file exists
                            os.mkdir(destDir)
                            destFile = os.path.join(destDir, f)
                            shutil.copyfile(filePath, destFile)
                            destTrfFile = os.path.join(destDir, "transformed_"+f[:-4]+'.csv')
                            shutil.copyfile(trfFile, destTrfFile)
                        else: # Replace the test file if the current one is smaller
                            currFSize = os.path.getsize(filePath)
                            for tstFile in os.listdir(destDir):
                                if not tstFile.startswith("transformed"):
                                    tstFilePath = os.path.join(destDir,tstFile)
                                    testFSize = os.path.getsize(tstFilePath)
                                    trfTstFilePath = os.path.join(destDir, "transformed_"+tstFile[:-4]+'.csv')
                                    if testFSize > currFSize:
                                        os.remove(tstFilePath)
                                        os.remove(trfTstFilePath)
                                        destFile = os.path.join(destDir, f)
                                        shutil.copyfile(filePath, destFile)
                                        destTrfFile = os.path.join(destDir, "transformed_"+f[:-4]+'.csv')
                                        shutil.copyfile(trfFile, destTrfFile)