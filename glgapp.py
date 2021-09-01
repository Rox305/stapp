import streamlit as st

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy_streamlit
import spacy
import os
# from PIL import Image
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt


# from tensorflow.keras.models import load_model
# model = tf.saved_model.load('use.h5')
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
# model = hub.load(module_url)

# st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Text Categorization")

st.write("""
*Finds topics in Medical/Tech news articles*
""")

db = pd.read_csv('base_clust.csv')
# vv = model(db['texto'])
# vv = np.load('vectores')
# nlp = spacy.load('en_core_web_sm')
# texto = ['Malaria has killed millions in Africa']

vectorizer = CountVectorizer(analyzer='word', max_features=2000)
X2 = vectorizer.fit_transform(db['article'])
matriz = X2.toarray()

texto = st.text_input("Enter text:")

if texto!='':
    a=vectorizer.transform([texto])
    distance_matrix = sklearn.metrics.pairwise.cosine_similarity(matriz, a)

    # distancias = sim_docs(model([texto]), vv)

    a = np.argsort(distance_matrix)

# maxi = st.number_input("Enter number of similar documents:",min_value=1, max_value=10, value=3, step=1)
    wtext=[]
    for i in range(1,6):
        wtext.append(db.iloc[a[-1]]['article'])
        # pp = round(db.iloc[a[-i]]['perc'],2)
        # if pp>.6:
        st.write(db.iloc[a[-i]]['descr'])
    # st.write(db.iloc[a[-i]]['subclas'])
        #  st.write(pp)
    # wtext = list(set(wtext))
    # wtext = ' '.join([str(t) for t in wtext])
    # word_cloud = WordCloud(collocations = False, background_color = 'white').generate(wtext)
    # plt.imshow(word_cloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()
    # st.pyplot()

st.title("Named Entity Recognition")
# docx = nlp(texto)
# spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)

