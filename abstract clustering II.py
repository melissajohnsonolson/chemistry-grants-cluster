# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:10:23 2019

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
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from gensim.models.phrases import Phrases, Phraser
import re
import datetime
import seaborn as sns

os.chdir('C:/Users/johns/Documents/Machine Learning/science funding project')
awds = pd.read_csv('NSF CHE 2015.csv', encoding='latin-1')
papers = pd.read_csv('che_paper_data.csv')


def program_clusters(pgms,n_topics,awds, papers):
    #First we need to filter the data by program code. Some grants have multiple program
    #codes, so we first filter through to determine which cells contain the program code
    #then we replace the exisiting program code(s) with the provided one. This ensures there
    #is only one code per award.
    papers = papers
    papers['year'] = pd.to_datetime(papers['year'])
    papers['citations per year'] = papers['citations'].divide(
        [((datetime.datetime.today()-x).days)/365.2422 for x in papers['year']])    
    num_pubs = papers.groupby('award number')[['publication']].count().reset_index()
    cits_year_mean = papers.groupby('award number')[['citations per year']].mean().reset_index()
    
    pgms = ['6878', '6880', '6882', '6883', '6884', '6885', '9101', '9102', '6881']
    awds = awds
    awds = awds[awds['ProgramElementCode(s)'].str.contains('|'.join(pgms))]
    for x in pgms:
        awds['ProgramElementCode(s)'] = np.where(awds['ProgramElementCode(s)'].str.contains(x), x, awds['ProgramElementCode(s)'] )
    awds['StartDate'] = pd.to_datetime(awds['StartDate'])
    awds['EndDate'] = pd.to_datetime(awds['EndDate'])
    awds['AwardedAmountToDate']=[x.replace('$', '') for x in awds['AwardedAmountToDate']]
    awds['AwardedAmountToDate']=[x.replace(',', '') for x in awds['AwardedAmountToDate']]
    awds['AwardedAmountToDate']=pd.to_numeric(awds['AwardedAmountToDate'])
    awds = pd.merge(awds, num_pubs, left_on='AwardNumber', right_on ='award number', how = 'left')
    awds = pd.merge(awds, cits_year_mean, left_on='AwardNumber', right_on ='award number', how = 'left')
    awds.drop(columns = ['award number_x', 'award number_y'], inplace = True)
    awds[['publication', 'citations per year']] = awds[[
            'publication', 'citations per year']].replace(np.nan, 0)
    awds['pubs per year'] = np.where(awds['EndDate']>datetime.datetime.today(), 
    awds['publication'].divide([((datetime.datetime.today()-x).days)/365.2422 for x in awds['StartDate']]), 
    awds['publication'].divide((awds['EndDate']-awds['StartDate']).astype('timedelta64[D]')/365.2422))



    
    abstracts = awds[['ProgramElementCode(s)', 'AwardNumber','Abstract', 'citations per year','pubs per year',
                      'AwardedAmountToDate']].copy()
    #This is a pretty clean data set, but there are some empty entries, so we
    #filter them out here
    abstracts = abstracts.dropna()
    
    #The first step in the tokenization process is splitting the abstract text
    #into a list of words.
    abstracts['clean_abstracts'] = [doc.lower().split() for doc in abstracts['Abstract']]
          
    #we want to account for possible bigrams and trigams, which we search for
    #here
    bigram = Phrases(list(abstracts['clean_abstracts']), min_count=5, threshold=20) 
    trigram =Phrases(bigram[list(abstracts['clean_abstracts'])], threshold=20)  
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)  
    
    #Now we start building our dictinary and creating the cleaned up corpus.
    #We start by creating a list of stop words, punctuation, and other text to remove.
    #we also instantiate a lemmatizer
    stop    = set(stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma   = WordNetLemmatizer()
    boiler_plate = 'This award reflects NSF''s statutory mission and has been deemed worthy of support through evaluation using the Foundation''s intellectual merit and broader impacts review criteria'
            
    #This function applies the bigram and trigram functions and lemmatizes the 
    #the abstracts and only keeps words that a greater than 2 characters
    def word_mod(doc):
        doc = re.sub('<.*?>', ' ', doc)
        doc = re.sub(boiler_plate, '', doc)
        punct_free  = ''.join(ch for ch in doc if ch not in exclude)
        words   = punct_free.lower().split()
        bigs = bigram_mod[words]
        tris = trigram_mod[bigs]
        stop_free  = " ".join([i for i in tris if i not in stop])
        lemm = " ".join(lemma.lemmatize(word) for word in stop_free.split())
        word_list = lemm.split()
        # only take words which are greater than 2 characters
        cleaned = [word for word in word_list if len(word) > 2]
        return cleaned
         
    
    abstracts['clean_abstracts'] = [word_mod(doc) for doc in abstracts['Abstract']]
    
    # Here we create the dictionary from the corpus of abstracts, where each unique term is assigned an index. 
    dictionary = corpora.Dictionary(abstracts['clean_abstracts'])
    # Filter terms which occurs in less than 4 articles & more than 40% of the abstracts 
    dictionary.filter_extremes(no_below=4, no_above=0.45)
    #This creates a sparse matrix of word frequencies in each abstracts
    abstract_term_matrix = [dictionary.doc2bow(doc) for doc in abstracts['clean_abstracts']]   
    
    # Here we create and train the LDA model, passing in our term frequncy matrix, the number of
    #topics/clusters to be created, and our dictionary
    ldamodel = Lda(abstract_term_matrix, num_topics= n_topics, id2word = dictionary, passes=50, iterations=500)
              
    # Here we print out the top 10 words for each topic and their weight
    for i,topic in enumerate(ldamodel.print_topics(num_topics=n_topics, num_words=10)):
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
    ab_hist = abstracts.groupby(['predicted topic'])['AwardNumber'].count()
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()] 
    cols = cols + cols + cols +cols
    f1, ax  = plt.subplots()
    ab_hist.plot.bar(rot = 0, color = cols )
    ax.set_xticklabels([x for x in ab_hist.index])
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
    
    topics = ldamodel.show_topics(formatted=False, num_topics = n_topics)
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
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, perplexity = 50, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)
    
    #Here we plot out the results for the t-SNE transformation
      
    mycolors = np.array(cols)
    
    title ="t-SNE Clustering of {} LDA Topics".format(n_topics)
    f = plt.figure()
    plt.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
    plt.title(title)
    plt.show()
    
    

    
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1,3,1)
    ax1.scatter(x = abstracts['AwardedAmountToDate'], y = abstracts['citations per year'], 
        color = mycolors[abstracts['predicted topic']])
    ax1.set_ylabel('Average Citations per Year')
    ax1.set_xlabel('Award Size [$]')
    ax1.set_title('Average Citiations per Year', fontsize = 11)
    ax2 = fig.add_subplot(1,3,2)
    ax2.scatter(x = abstracts['AwardedAmountToDate'], y = abstracts['pubs per year'], 
        color = mycolors[abstracts['predicted topic']])
    ax2.set_ylabel('Number Publications per Year')
    ax2.set_xlabel('Award Size [$]')
    ax2.set_title('Number of Publications per Year', fontsize = 11)
    ax3 = fig.add_subplot(1,3,3)
    ax3.scatter(x = abstracts['pubs per year'], y = abstracts['citations per year'],
        color = mycolors[abstracts['predicted topic']])
    ax3.set_xlabel('Number Publications per Year')
    ax3.set_ylabel('Average Citiations per Year')
    ax3.set_title('Number Publications vs \nAverage Citation Count', fontsize = 11)    
    from matplotlib.legend_handler import HandlerPatch
    class HandlerEllipse(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            p = mpatches.Ellipse(xy=center, width=height + xdescent,
                                 height=height + ydescent)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]
    handles = [ mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor=mycolors[i], edgecolor="none" ) for i in range(0,n_topics)]
    handles =  [mpatches.Circle((0.5, 0.5), radius =0.25, facecolor='w', edgecolor="none" )]+handles    
    legend_labels = list(range(0,n_topics))
    legend_labels = ['Topic'] + legend_labels
    ax3.legend(handles, legend_labels, bbox_to_anchor=(1, .88),  bbox_transform=fig.transFigure,
               handler_map={mpatches.Circle: HandlerEllipse()})
    plt.tight_layout()
    
pgm_list = ['6878', '6880', '6882', '6883', '6884', '6885', '9101', '9102', '6881']

program_clusters([ '9101', '9102'],15, awds, papers)
