#!/bin/bash

# Use gdown to download the google link
source ../env/bin/activate
pip install gdown
gdown https://drive.google.com/u/1/uc?id=1LIbAyaml39ahmyLpRTBYwEY_5rnoSMp9
deactivate

# Remove tar file
tar -xf data.tar.gz
sleep 5
rm data.tar.gz
