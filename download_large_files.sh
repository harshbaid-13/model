#!/bin/bash
SUBDIR="./models"

# Create the subdirectory if it doesn't exist
mkdir -p $SUBDIR

echo "Downloading large files to $SUBDIR directory..."

# Use gdown to download files
gdown --output $SUBDIR/bert_model_100.pkl "https://drive.google.com/uc?id=11V4em_xD8PAkIUgyrzJkcd4RTfWzud0N"
gdown --output $SUBDIR/sentiment_analyzer.pkl "https://drive.google.com/uc?id=17NBhHJAhJWr1ikUsUUgjhmLNbbS6fu8q"

echo "Download completed."
