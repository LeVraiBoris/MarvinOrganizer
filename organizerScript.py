import os
from posixpath import basename
import re
from zipfile import ZipFile

import marvinOrganizer as marv
import argparse
import sys
import numpy as np
import shutil
import pandas as pd
DDEBUG = True
USE_GOOEY = True
if USE_GOOEY is True:
    from gooey import Gooey, GooeyParser # Use GUI
UPDATE_MODEL = True
CST_MIN_CONFIDENCE = 0.0

def unzipInPlace(inRootPath:str, marvin:marv.MarvinOrganizer):
    """Go down the directory tree and unpack every zip archive that is found.

    Args:
        inRootPath (str): path to the folder to sort
        marvin (marv.MarvinOrganizer): an instance of the Marvin organizer 
    """
    for dirPath, dList, fList in os.walk(inRootPath):
        for f in fList:
            split = f.split(".")
            ext = split[-1]
            fname = split[0]
            if len(fname) > 0 and ext in marv.CST_ZIP_EXTENSIONS: # This will ignore hidden files (on an Unix like system)
                zipDir = os.path.join(dirPath, "Zips")
                if not os.path.exists(zipDir):
                    os.mkdir(zipDir)
                fPath = os.path.join(dirPath, f)
                with ZipFile(fPath) as arch:
                    arch.extractall(zipDir)
                    # We go as far as searching for zips inside zips
                    for f in os.listdir(zipDir):
                        split = f.split(".")
                        ext = split[-1]
                        fname = split[0]
                        if len(fname) > 0 and ext in marv.CST_ZIP_EXTENSIONS: # This will ignore hidden files (on an Unix like system)
                            fPath = os.path.join(zipDir, f)
                            with ZipFile(fPath) as arch:
                                arch.extractall(zipDir)

def sortFolder(inRootPath:str, outRootPath:str, marvin:marv.MarvinOrganizer, consolidate=True):
    reportFile = os.path.join(outRootPath ,"marvinOrganizerReport.csv")
    reportDct = {"OriginalPath":[], "Filename":[], "FileType":[], "Source":[], "Score":[], "SortedPath":[],}
    print("Sorting..", end=" ")
    # Iterate over files in directory
    for dirPath, dList, fList in os.walk(inRootPath):
        # Open file
        for f in fList:
            fPath = os.path.join(dirPath, f)
            ext = marvin.utils.normalizeString(f.split(".")[-1])
            if ext in marv.CST_TAB_EXTENSIONS:
                fileType = "Tabular"
            elif ext in marv.CST_PDF_EXTENSIONS:
                fileType = "Pdf"
            else:
                fileType = "Other"
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

            destFile = os.path.join(destDir, f)
            if os.path.exists(destFile):
                destDir = os.path.join(outRootPath,"Duplicates")
                destFile = os.path.join(destDir, f)
            if not os.path.isdir(destDir):
                os.mkdir(destDir)
            shutil.copyfile(fPath, destFile)
            reportDct["OriginalPath"].append(fPath)
            reportDct["Filename"].append(f)
            reportDct['FileType'].append(fileType)
            reportDct['Source'].append(sourceLabel)
            reportDct["Score"].append(score)
            reportDct['SortedPath'].append(destDir)

    reportDF = pd.DataFrame.from_dict(reportDct)
    # Propagate results
    consolidatedDF =  marvin.extendClassification()
    consolidatedDF.index = reportDF.index
    if consolidate is True: # Make the output of the consolidaton the official output
        reportDF[['Source']] = consolidatedDF[['sourcesList']]
        reportDF[['Score']] = consolidatedDF[['sourcesProba']]
    else:
        reportDF[['SourceInfered']] = consolidatedDF[['sourcesList']]
        reportDF[['SourceInferedScore']] = consolidatedDF[['sourcesProba']]
    reportDF.to_csv(reportFile)

def retrainMarvin(organizedPath:str, marvin:marv.MarvinOrganizer, updateModel=False):
    reportFile = os.path.join(organizedPath ,"marvinOrganizerReport.csv")
    #reportDct = {"OriginalPath":[], "SortedPath":[], "Source":[], "Score":[]}
    trainDataDF = pd.read_csv(reportFile)
    duplicatesPath = os.path.join(organizedPath, "Duplicates")
    print("Retraining..", end=" ")

    # Iterate over the report and fixed the directory tree according to the indicated source in the report file
     # Iterate over the report and fixedd the directory tree according to the indicated source in the report file
    for idx, r in trainDataDF.iterrows():
        desiredFolder = r["Source"]
        fileType = r["FileType"]
        print(organizedPath," + ", desiredFolder)
        if fileType == "Tabular":
            desiredPath = os.path.join(organizedPath, desiredFolder)
        else:
            desiredFolder = desiredFolder + " " + fileType
            desiredPath = os.path.join(organizedPath, desiredFolder)
        realPath = r["SortedPath"]
        if realPath != duplicatesPath:
            if desiredPath != realPath:
                # Move the file 
                srcFile = os.path.join(realPath, r["Filename"])
                destFile = os.path.join(desiredPath, r["Filename"])
                if updateModel is True:
                    marvin.updateFromFile(srcFile, r["Source"])
                if not os.path.isdir(desiredPath):
                    os.mkdir(desiredPath)
                if os.path.exists(destFile) is not True:
                    os.rename(srcFile, destFile)
                trainDataDF.loc[idx, "SortedPath"] = desiredPath
    trainDataDF.to_csv(reportFile, index=False)
    if updateModel is True: # Only save the model when not debuggging
        marvin.toJson()

def run(args):
    version = marv.CST_MODEL_VERSION
    modelPath = "./Model/"
    saveModelName = 'marvinOrganizer_'+"_FastTxt"+version
    loadModelName = 'marvinOrganizer_'+"_FastTxt"+version
    signatureModelName = 'marvinOrganizerSignatures_'+marv.CST_VCT_TYPE+version+".json"
    utils = marv.MarvinOrganizerUtils()
    marvin = marv.MarvinOrganizer()
    if os.path.exists(modelPath+signatureModelName): # load the model
        marvin.fromJson(modelPath+signatureModelName)
        print("Model Loaded..")
    else: 
        print("Marvin model not found, exiting.")
        return

    inRootPath = args.input_folder
    outRootPath = args.output_folder
    retrain = args.fix
    unzip = args.unzip_all
    consolidate = args.consolidate

    if DDEBUG is True:  # When debugging  delete any existing output
        deleteOutPath = True
    else:
        deleteOutPath = args.delete_existing

    if outRootPath == "Default":
        if retrain is False:
            basePath = os.path.dirname(inRootPath) 
            outFolderName = os.path.basename(inRootPath) + ' Organized'
            outRootPath = os.path.join(basePath, outFolderName) 
        else:
            if os.path.exists(os.path.join(inRootPath,'marvinOrganizerReport.csv')):
                outRootPath = inRootPath
            else:
                basePath = os.path.dirname(inRootPath) 
                outFolderName = os.path.basename(inRootPath) + ' Organized'
                outRootPath = os.path.join(basePath, outFolderName) 

    if retrain is False:
        if unzip is True:
            unzipInPlace(inRootPath, marvin)
        if not os.path.isdir(outRootPath):
            os.mkdir(outRootPath)
        elif deleteOutPath is True:
            shutil.rmtree(outRootPath)
            os.mkdir(outRootPath)
    else:
        if not os.path.isdir(outRootPath):
            print("Organized folder not found. Exiting..")
            return

    print("In path: ", inRootPath)
    print("Out path: ", outRootPath)
    
    if retrain is False:
        # Sort the input folder 
        sortFolder(inRootPath, outRootPath, marvin, consolidate = consolidate)
        if consolidate is True: # Automatically fix but do not update the model
            retrainMarvin(outRootPath, marvin, updateModel=False)
    else:
        # Update Marvin according to the instructions given in "marvinOrganizerReport.csv"
        retrainMarvin(outRootPath, marvin, updateModel=UPDATE_MODEL)
    print("done !")
    return

def main():
    if USE_GOOEY is False:
        cmdParser = argparse.ArgumentParser(description='Sort statement files according to RYLTY source .')
        cmdParser.add_argument("input_folder", 
                                help="Folder to sort (aka $inputDir)")
        cmdParser.add_argument("-o", "--output_folder", 
                                help="output folder for the sorted files (Optionnal). Defaults to $rootFolder +\' Organized\' (aka $outputDir)",
                                default="Default")
    else:
        cmdParser = GooeyParser(description='Sort statement files according to RYLTY source .')
        cmdParser.add_argument("input_folder", 
                                widget='DirChooser',
                                help="Folder to sort (aka $inputDir)")
        cmdParser.add_argument("-o", "--output_folder", 
                                widget='DirChooser',
                                help="output folder for the sorted files (Optionnal). Defaults to $rootFolder +\' Organized\' (aka $outputDir)",
                                default="Default")
    cmdParser.add_argument("-d", "--delete_existing",
                    help="Delete pre-existing output folder (default) unless \'--fix\' is checked",
                    action='store_true')
    cmdParser.add_argument("-c", "--consolidate",
                    help="Consolidate the classification based on folder structure inference",
                    action='store_true')
    cmdParser.add_argument("-f", "--fix",
                    help="Fix the mistakes in the Organized file system based on sources given in  $outputDir/marvinOrganizerReport.csv",
                    action='store_true')
    cmdParser.add_argument("-z", "--unzip_all",
                    help="Unzip all archives found in $inputDir before sorting",
                    action='store_true')
    args = cmdParser.parse_args()
    run(args)

if __name__ == "__main__":
    if USE_GOOEY is False: # CLI only 
        main()
    else:
        main = Gooey(main)
        main()