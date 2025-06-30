#!/bin/bash

# Set the source directory where the .wav files are
SOURCE_DIR="/root/khanhnnm/se/LibriMix/Libri2Mix/wav16k/min/train-100/noisy"

# Create 5 subdirectories
for i in {1..20}
do
  mkdir -p "$SOURCE_DIR/folder$i"
done

# Get a list of all .wav files and distribute them randomly into the 5 folders
FILES=($SOURCE_DIR/*.wav)
TOTAL_FILES=${#FILES[@]}
NUM_FOLDERS=20

for ((i=0; i<TOTAL_FILES; i++))
do
  # Get the current file
  FILE=${FILES[$i]}

  # Calculate which folder the file will go to
  FOLDER_NUM=$((i % NUM_FOLDERS + 1))

  # Move the file to the corresponding folder
  mv "$FILE" "$SOURCE_DIR/folder$FOLDER_NUM/"
done

echo "Files have been distributed into 20 folders."