#!/bin/bash
PROGRAM_NAME='organizerScript'
echo  'Building '$PROGRAM_NAME
LOCAL_WINE_STORE=~/.wine/drive_c/users/boris/AppData/Local/Programs
WIN_PIP=$LOCAL_WINE_STORE/Python/Python310/Scripts/pip3.exe
WIN_PYTHON=$LOCAL_WINE_STORE/Python/Python310/python.exe

WIN_INSTALLER=$LOCAL_WINE_STORE/Python/Python310/Scripts/pyinstaller.exe
echo 'Cleaning up...' 
rm ./dist -rf
echo 'Update requirements...'
wine $WIN_PIP install -r requirements.txt
echo 'Building package...'
wine $WIN_INSTALLER $PROGRAM_NAME.py

cp -r ./Model ./dist/$PROGRAM_NAME
echo 'Building archive...'
zip -r ./$PROGRAM_NAME.zip ./dist/$PROGRAM_NAME