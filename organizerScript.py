import os
import re
import marvinOrganizer as marv
import argparse
import sys
import numpy as np
import shutil
import pandas as pd

DDEBUG = False
CST_MIN_CONFIDENCE = 0.0

def sortFolder(inRooPath:str, outRootPath:str, marvin:marv.MarvinOrganizer):
    reportFile = os.path.join(outRootPath ,"marvinOrganizerReport.csv")
    reportDct = {"OriginalPath":[], "Filename":[], "Source":[], "Score":[], "SortedPath":[],}
    
    # Iterate over files in directory
    for dirPath, dList, fList in os.walk(inRootPath):
        # Open file
        for f in fList:
            fPath = os.path.join(dirPath, f)
            source, sourceScore = marvin.getSources(fPath)
            sourceLabel="Unsorted"
            score = -1
            if len(source) > 0:
                sourceIdx = np.argmax(sourceScore)
                if sourceScore[sourceIdx] > CST_MIN_CONFIDENCE:
                    destDir = os.path.join(outRootPath, source[sourceIdx])
                    sourceLabel = source[sourceIdx]
                    score = sourceScore[sourceIdx] 
                else:
                    destDir = os.path.join(outRootPath, "Unsorted")
                    sourceLabel = "Unsorted"
                    score = 0
            else:
                destDir = os.path.join(outRootPath, "Unsorted")

            if not os.path.isdir(destDir):
                os.mkdir(destDir)
            destFile = os.path.join(destDir, f)
            shutil.copyfile(fPath, destFile)
            reportDct["OriginalPath"].append(fPath)
            reportDct["Filename"].append(f)
            reportDct['Source'].append(sourceLabel)
            reportDct["Score"].append(score)
            reportDct['SortedPath'].append(destDir)

        reportDF = pd.DataFrame.from_dict(reportDct)
        reportDF.to_csv(reportFile)

def retrainMarvin(organizedPath:str, marvin:marv.MarvinOrganizer):
    reportFile = os.path.join(organizedPath ,"marvinOrganizerReport.csv")
    #reportDct = {"OriginalPath":[], "SortedPath":[], "Source":[], "Score":[]}
    trainDataDF = pd.read_csv(reportFile)
    # Iterate over the report and fixedd the directory tree according to the indicated source in the report file
    for idx, r in trainDataDF.iterrows():
        desiredFolder = r["Source"]
        desiredPath = os.path.join(organizedPath, desiredFolder)
        realPath = r["SortedPath"]
        if desiredPath != realPath:
            # Move the file 
            srcFile = os.path.join(realPath, r["Filename"])
            destFile = os.path.join(desiredPath, r["Filename"])
            marvin.updateFromFile(srcFile, r["Source"])
            trainDataDF.loc[idx, "SortedPath"] = desiredPath
            if not os.path.isdir(desiredPath):
                os.mkdir(desiredPath)
            os.rename(srcFile, destFile)
    trainDataDF.to_csv(reportFile)

    if DDEBUG is False:
        marvin.toJson()

if __name__ == "__main__":
    version = marv.CST_MODEL_VERSION
    modelPath = "./Model/"
    saveModelName = 'marvinOrganizer_'+"_FastTxt"+version
    loadModelName = 'marvinOrganizer_'+"_FastTxt"+version
    signatureModelName = 'marvinOrganizerSignatures_'+marv.CST_VCT_TYPE+version+".json"
    utils = marv.MarvinOrganizerUtils()
    marvin = marv.MarvinOrganizer()
    if os.path.exists(modelPath+signatureModelName): # load the model
        marvin.fromJson(modelPath+signatureModelName)
    else: # Build the dataset 
        print("Marvin model not found, exiting.")
        pass

    arguments = sys.argv
    cmdParser = argparse.ArgumentParser(description='Sort statement files according to RYLTY source .')
    cmdParser.add_argument("input_folder", 
                        type=str, 
                        help="root folder to sort")
    cmdParser.add_argument("-o", "--output_folder", 
                        type=str, 
                        help="root folder where to store the sorted files (Optionnal). Defaults to $rootFolder +\' Organized\'",
                        default="Default")
    cmdParser.add_argument("-r", "--retrain",
                        help="Fix the mistakes in the Organized file system based on sources given in  $rootFolder +\' Organized\'/marvinOrganizerReport.csv",
                        action='store_true')
    args = cmdParser.parse_args()
    inRootPath = args.input_folder
    outRootPath = args.output_folder
    retrain = args.retrain
    if outRootPath == "Default":
        outRootPath = inRootPath[:-1] if inRootPath[-1]=="/" else inRootPath
        outRootPath = outRootPath+" Organized"

    if not os.path.isdir(outRootPath):
        if retrain is True:
            print("Organized folder not found. Exiting..")
            pass
        else:
            os.mkdir(outRootPath)

    if retrain is False:
        # Sort the input folder 
        sortFolder(inRootPath, outRootPath, marvin)
    else:
        # Update Marvin according to the instructions given in "marvinOrganizerReport.csv"
        retrainMarvin(outRootPath, marvin)