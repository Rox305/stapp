import streamlit as st

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

import spacy_streamlit
import spacy
import os
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt


from tensorflow.keras.models import load_model
# model = tf.saved_model.load('use.h5')
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Text Categorization")

st.write("""
*Finds topics in Medical/Tech news articles*
""")

db = pd.read_csv('Data_clusters_sin.csv')
vv = model(db['texto'])
# vv = np.load('vectores')
nlp = spacy.load('en_core_web_sm')
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

if texto!='':
    distancias = sim_docs(model([texto]), vv)

    a = np.argsort(distancias)

# maxi = st.number_input("Enter number of similar documents:",min_value=1, max_value=10, value=3, step=1)
    wtext=[]
    for i in range(1,6):
        wtext.append(db.iloc[a[-1]]['texto'])
        pp = round(db.iloc[a[-i]]['perc'],2)
        if pp>.6:
         st.write(db.iloc[a[-i]]['clas'], db.iloc[a[-i]]['subclas'])
    # st.write(db.iloc[a[-i]]['subclas'])
         st.write(pp)
    wtext = list(set(wtext))
    wtext = ' '.join([str(t) for t in wtext])
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(wtext)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()

st.title("Named Entity Recognition")
docx = nlp(texto)
spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)

