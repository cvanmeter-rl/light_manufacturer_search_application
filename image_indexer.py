import pandas as pd
import glob
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer
import pickle
from io import BytesIO

#load clip model
model = SentenceTransformer('clip-ViT-B-32')

#load excel files
all_files = glob.glob("data/*.xlsx")
