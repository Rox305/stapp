import streamlit as st

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

import en_core_web_sm
import spacy_streamlit
import spacy
import os
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import en_core_web_sm

nlp = en_core_web_sm.load()


from tensorflow.keras.models import load_model
# model = tf.saved_model.load('use.h5')
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)


st.title("Text Categorization")

st.write("""
*Finds topics in Medical/Tech news articles*
""")

db = pd.read_csv('Data_clusters_sin.csv')
# vv = model(db['texto'])
# vv = np.load('vectores')
# nlp = spacy.load('en_core_web_sm')
# nlp = en_core_web_sm.load()

# texto = ['Malaria has killed millions in Africa']


def angular_similarity(emb1, emb2):
  cos_sim = np.inner(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
  ang_dist = np.arccos(cos_sim)/np.pi
  return 1-ang_dist

def sim_docs(text, data, th=.5):
  md = np.zeros(len(data))
  for i in range(len(data)):
    md[i] = angular_similarity(text, data[i])
  # sims=[h in md for h in md if h>th]
  return md

texto = st.text_input("Enter text:")

st.title("Named Entity Recognition")
docx = nlp(texto)
spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)

