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
from organizerScript import run
UPDATE_MODEL = True
CST_MIN_CONFIDENCE = 0.0
# CST_TAB_EXTENSIONS = ['csv', 'xl', 'xls', 'xlsx', 'txt', 'tab', 'xlsb']
# CST_PDF_EXTENSIONS = ['pdf']
# CST_ZIP_EXTENSIONS = ['zip', 'gzip']

if __name__ == "__main__":

    cmdParser = argparse.ArgumentParser(description='Sort statement files according to RYLTY source .')
    cmdParser.add_argument("input_folder", 
                            help="Folder to sort (aka $inputDir)")
    cmdParser.add_argument("-o", "--output_folder", 
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

