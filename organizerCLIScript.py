
import os
import zipfile
from zipfile import ZipFile

import marvinOrganizer as marv
import argparse
import numpy as np
import shutil
import pandas as pd
import organizerUtils
DDEBUG = False

def main():
    cmdParser = argparse.ArgumentParser(description='Sort statement files according to RYLTY source .')
    cmdParser.add_argument("input_folder", 
                            help="Folder to sort (aka $inputDir)")
    cmdParser.add_argument("-o", "--output_folder", 
                            help="output folder for the sorted files (Optionnal). Defaults to $rootFolder +\' Organized\' (aka $outputDir)",
                            default="Default")
    cmdParser.add_argument("-d", "--delete_existing",
                    help="Delete pre-existing output folder (default) unless \'--fix\' is checked",
                    action='store_true')
    cmdParser.add_argument("-z", "--unzip_all",
                    help="Unzip all archives found in $inputDir before sorting",
                    action='store_true')
    cmdParser.add_argument("-c", "--consolidate",
                    help="Consolidate the classification based on folder structure inference",
                    action='store_true')
    cmdParser.add_argument("-f", "--fix",
                    help="Fix the mistakes in the Organized file system based on sources given in  $outputDir/marvinOrganizerReport.csv",
                    action='store_true')
    cmdParser.add_argument("-u", "--update_model",
                    help="Update the organizer model when fixing misakes",
                    action='store_true')
    args = cmdParser.parse_args()
    organizerUtils.run(args)

if __name__ == "__main__":
    # Run CLI
    main()