import os
import marvinOrganizer as marv
import argparse
import sys
import numpy as np
import shutil
import pandas as pd

CST_MIN_CONFIDENCE = 0.0
if __name__ == "__main__":
    # Select task
    marvinMatchTest = True
    trainFastTxt = False
    version = marv.CST_MODEL_VERSION
    modelPath = "./Model/"
    saveModelName = 'marvinOrganizer_'+"_FastTxt"+version
    loadModelName = 'marvinOrganizer_'+"_FastTxt"+version
    signatureModelName = 'marvinOrganizerSignatures_'+marv.CST_VCT_TYPE+version+".json"
    utils = marv.MarvinOrganizerUtils()
    marvin = marv.MarvinOrganizer()

    arguments = sys.argv
    cmdParser = argparse.ArgumentParser(description='Sort statement files according to RYLTY source .')
    cmdParser.add_argument("-i", "--input_folder", 
                        type=str, 
                        help="root folder to sort")
    cmdParser.add_argument("-o", "--output_folder", 
                        type=str, 
                        help="root folder where to store the sorted files",
                        default="Default")
    
    args = cmdParser.parse_args()
    inRootPath = args.input_folder
    outRootPath = args.output_folder
    if outRootPath == "Default":
        outRootPath = inRootPath[:-1] if inRootPath[-1]=="/" else inRootPath
        outRootPath = outRootPath+" Organized"
    if not os.path.isdir(outRootPath):
        os.mkdir(outRootPath)

    reportFile = os.path.join(outRootPath ,"marvinOrganizerReport.csv")
    reportDct = {"OriginalPath":[], "SortedPath":[], "Source":[], "Score":[]}
    if os.path.exists(modelPath+signatureModelName): # load the model
        marvin.fromJson(modelPath+signatureModelName)
    else: # Build the dataset 
        print("Marvin model not found, exiting.")
        pass

    # Iterate over files in directory
    for dirPath, dList, fList in os.walk(inRootPath):
        # Open file
        for f in fList:
            fPath = os.path.join(dirPath, f)
            source, sourceScore = marvin.getSources(fPath)
            sourceLabel="N/A"
            score = -1
            if len(source) > 0:
                sourceIdx = np.argmax(sourceScore)
                if sourceScore[sourceIdx] > CST_MIN_CONFIDENCE:
                    destDir = os.path.join(outRootPath, source[sourceIdx])
                    sourceLabel = source[sourceIdx]
                    score = sourceScore[sourceIdx] 
                else:
                    destDir = os.path.join(outRootPath, "Unsorted")
                    sourceLabel = "Unknown"
                    score = 0
            else:
                destDir = os.path.join(outRootPath, "Unsorted")

            if not os.path.isdir(destDir):
                os.mkdir(destDir)
            destPath = os.path.join(destDir, f)
            shutil.copyfile(fPath, destPath)
            reportDct["OriginalPath"].append(fPath)
            reportDct['Source'].append(sourceLabel)
            reportDct['SortedPath'].append(outRootPath)
            reportDct["Score"].append(score)

        reportDF = pd.DataFrame.from_dict(reportDct)
        reportDF.to_csv(reportFile)