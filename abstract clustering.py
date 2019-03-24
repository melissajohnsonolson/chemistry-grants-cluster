# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:30:24 2019

@author: molson
"""

import os
import operator
import pandas as pd
import numpy as np
from gensim import corpora
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim.models.ldamodel import LdaModel as Lda
from nltk.stem.snowball import SnowballStemmer
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from gensim.models.phrases import Phrases, Phraser
import re
os.chdir('C:/Users/johns/Documents/Machine Learning/science funding project')




def program_clusters(pgms,n_topics,data):
    #First we need to filter the data by program code. Some grants have multiple program
    #codes, so we first filter through to determine which cells contain the program code
    #then we replace the exisiting program code(s) with the provided one. This ensures there
    #is only one code per award.
    awds = data
    awds = awds[awds['ProgramElementCode(s)'].str.contains('|'.join(pgms))]
    for x in pgms:
        awds['ProgramElementCode(s)'] = np.where(awds['ProgramElementCode(s)'].str.contains(x), x, awds['ProgramElementCode(s)'] )
        
    abstracts = awds[['ProgramElementCode(s)', 'AwardNumber','Abstract']].copy()
    #This is a pretty clean data set, but there are some empty entries, so we
    #filter them out here
    abstracts = abstracts.dropna()
    
    #Here we start building our dictinary and creating the cleaned up corpus.
    #We start by  removing stop words, punctuation, and stemming or lemmatizing
    #he abstract text
    stop    = set(stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma   = WordNetLemmatizer()
    stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
    # pass the article text as string "doc"
    
    #Here we use a small nexted function to pass through each abstract individually
    def clean(doc):
        #here we clean up errent breaks like <br/>
        doc = re.sub('<.*?>', ' ', doc)
        #This creates a long string
        #of words while excluding stop words
        stop_free  = " ".join([i for i in doc.lower().split() if i not in stop])
        #This goes through each character and removes punctuation
        punct_free  = ''.join(ch for ch in stop_free if ch not in exclude)
        words   = punct_free.split()
        return words
        
    
    #Here is where we pass each abstract through the cleaning function
    abstracts['clean_abstracts'] = [clean(doc) for doc in abstracts['Abstract']]
    
    # So we can use bigrams and trigrams, we create new models, running through our
    #cleaned abstracts
    bigram = Phrases(list(abstracts['clean_abstracts']), min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram =Phrases(bigram[list(abstracts['clean_abstracts'])], threshold=100)  
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)  
    
    #This function applies the bigram and trigram functions and lemmatizes the 
    #the abstracts and only keeps words that a greater than 2 characters
    def word_mod(doc):
        bigs = bigram_mod[doc]
        tris = trigram_mod[bigs]
        lemm = " ".join(lemma.lemmatize(word) for word in tris)
        #stemm    = " ".join(stemmer2.stem(word) for word in punct_free.split())
        words = lemm.split()
        # only take words which are greater than 2 characters
        cleaned = [word for word in words if len(word) > 2]
        return cleaned
    abstracts['clean_abstracts'] = [word_mod(doc) for doc in abstracts['clean_abstracts']]  
    
    
    # Here we create the dictionary from the corpus of abstracts, where each unique term is assigned an index. 
    dictionary = corpora.Dictionary(abstracts['clean_abstracts'])
    # Filter terms which occurs in less than 4 articles & more than 40% of the abstracts 
    dictionary.filter_extremes(no_below=4, no_above=0.4)
    #This creates a sparse matrix of word frequencies in each abstracts
    abstract_term_matrix = [dictionary.doc2bow(doc) for doc in abstracts['clean_abstracts']]   
    
    # Here we create and train the LDA model, passing in our term frequncy matrix, the number of
    #topics/clusters to be created, and our dictionary
    ldamodel = Lda(abstract_term_matrix, num_topics= n_topics, id2word = dictionary, passes=15, iterations=500)
              
    # Here we print out the top 10 words for each topic and their weight
    for i,topic in enumerate(ldamodel.print_topics(num_topics=10, num_words=10)):
       words = topic[1].split("+")
       print (words,"\n")
     
     #Next we want to know what topic each abstract belongs to we pass each abstract
     #into the get_document_topics method and it returns the topic and the 
     #probability of the abstract beloning to a that topic. We take the one that
     #has the highest probability
    def pred_topic(doc):
        doc_bow = ldamodel.id2word.doc2bow(doc)
        doc_topics = ldamodel.get_document_topics(doc_bow, minimum_probability=0.20)  
        if doc_topics:
            doc_topics.sort(key = operator.itemgetter(1), reverse=True)
            theme = doc_topics[0][0]
        else:
            theme = np.nan
        return theme

    abstracts['predicted topic'] = [pred_topic(doc) for doc in abstracts['clean_abstracts']]
    
    #Here we do a histogram of how many abstracts/awards fall into each topic
    ab_hist = abstracts.groupby(['predicted topic','ProgramElementCode(s)'])['AwardNumber'].count()
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()] 
    f1, ax  = plt.subplots()
    ab_hist.plot.bar(rot = 0, color = cols )
    ax.set_xticklabels([x[0] for x in ab_hist.index])
    ax.set_xlabel('Topic Number')
    ax.set_ylabel('Count of Awards in Topic')
    ax.set_title('Distribution of Awards in Derived Topic Areas')
    plt.show()
    
    #Here we create a word cloud for each of the top words in the topic. Their size 
    #is indicative of their weight.
    cloud = WordCloud(stopwords=stopwords.words('english'),
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)
    
    topics = ldamodel.show_topics(formatted=False)
    fig, axes = plt.subplots(1, n_topics, figsize=(10,10), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')   
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    
        
    #Next we'll do a t-SNE plot clustering the abstracts based off the topic
    #probabilities returned from the model. This creates a array where each
    #column is a topic and each row is an abstract and each entry is the probability
    #that the abstract belongs to that topic.
    col_ns = range(0,n_topics)
    topic_weights = pd.DataFrame(columns = col_ns)
    for i in range(0,len(ldamodel[abstract_term_matrix])):
        weights = ldamodel[abstract_term_matrix][i]
        for j in range(0, len(weights)):
           entry = pd.DataFrame(columns = col_ns)
           idx = weights[j][0]
           entry.loc[0,idx] = weights[j][1]
        topic_weights = topic_weights.append(entry)
    topic_weights.reset_index(drop = True, inplace = True)
    
    # Replace any nan entries (because there was zero probability the 
    #abstract belonged in that topic) with zero
    arr = pd.DataFrame(topic_weights).fillna(0).values
    
    # We can limit this to only well separated abstracts as well
    #arr = arr[np.amax(arr, axis=1) > 0.15]
    
    # This is pulls out the highest probability topic for each abstract.  We'll
    #use this for the color scheme in the t-SNE plot.
    topic_num = np.argmax(arr, axis=1)
    
    # Here we initialize and fit our t-SNE model
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)
    
    #Here we plot out the results for the t-SNE transformation
      
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    title ="t-SNE Clustering of {} LDA Topics".format(n_topics)
    f = plt.figure()
    plt.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
    plt.title(title)
    plt.show()

        

pgm_list = ['6878', '6880', '6882', '6883', '6884', '6885', '9101', '9102', '6881']
awds = pd.read_csv('NSF CHE 2015.csv', encoding='latin-1')
program_clusters(['6884'], 6, awds)

