#!/bin/bash
PROGRAM_NAME='organizerScript'
echo $PROGRAM_NAME
LOCAL_WINE_STORE=~/.wine/drive_c/users/boris/AppData/Local/Programs
WIN_PIP=$LOCAL_WINE_STORE/Python/Python310/Scripts/pip3.exe
echo $WIN_PIP
WIN_PYTHON=$LOCAL_WINE_STORE/Python/Python310/python.exe
echo $WIN_PYTHON
WIN_INSTALLER=$LOCAL_WINE_STORE/Python/Python310/Scripts/pyinstaller.exe
rm ./dist -rf
wine $WIN_PIP install -r requirements.txt
wine $WIN_INSTALLER $PROGRAM_NAME.py

cp -r ./Model ./dist/$PROGRAM_NAME
zip -r ./$PROGRAM_NAME.zip ./dist/$PROGRAM_NAME