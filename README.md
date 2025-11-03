# Environment Setup
1. python -m venv venv
2. ./venv/Scripts/activate   (if venv not already activated)
3. pip install -r requirements.txt   (if not already installed)


# How to Access Datasets:

### BEA-2019 
 - Go to https://www.cl.cam.ac.uk/research/nl/bea2019st/?utm_source=chatgpt.com#data
 - Click on the "Data" tab then scroll down to "Other Corpora and Download Links"
 - Download W&I+LOCNESS v2.1 (this is BEA-2019)
 - run python -m spacy download en_core_web_sm
 - The rest is done for you in datasets.ipynb

### NUCLE
 - Request from https://www.comp.nus.edu.sg/~nlp/corpora.html

### JFLGE (CONNL Replacement)
 - Loaded in datasets.ipynb

### WikiAuto
 - Loaded in datasets.ipynb

### ASSET
 - Loaded in datatesets.ipynb
