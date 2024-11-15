#!/bin/bash
echo "Downloading large files to /tmp directory..."

wget -O /tmp/bert_model_100.pkl "https://iitk-my.sharepoint.com/:u:/g/personal/bharsh24_iitk_ac_in/EWodu-uW70xBjDZxYPLGMA8B2Va8Yfejicp07PBRnq6xaQ?e=efDjjz"
wget -O /tmp/sentiment_analyzer.pkl "https://iitk-my.sharepoint.com/:u:/g/personal/bharsh24_iitk_ac_in/ERdyfsAJOg5FtCzWchq8kg8Bbt-YdJ5kDdUapHXarLtcLQ?e=mnimpJ"

echo "Download completed."
