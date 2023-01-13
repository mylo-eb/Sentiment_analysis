import pandas as pd
import csv
import re 
import string
import matplotlib.pyplot as plt
df=pd.read_csv(r'C:\Users\myloe\OneDrive\Desktop\combined.csv')

#removing URls 
import re
def url_remove(textur):
    textunur=re.sub(r"\S*https?:\S*", "", textur)
    return textunur
df.text = df.text.apply(lambda x: url_remove(x))


#Fixing Contractions
import contractions
def fix_contract(textcont):
    textfix=contractions.fix(textcont)
    return textfix
df.text=df.text.apply(lambda x: fix_contract(x))


# remove hashtags
# only removing the hash # sign from the word
import re
def hash_remove(texthash):
    textunhash=re.sub(r'#', '', texthash)
    return textunhash
df.text = df.text.apply(lambda x: hash_remove(x))


#dictionary consisting of apostrphe's
import re
def s_remove(texts):
    textuns=re.sub(r"'s", '', texts)
    return textuns
df.text = df.text.apply(lambda x: s_remove(x))



#dictionary consisting of apostrphe’s is

import re
def is_fix(textunis):
    textis=re.sub(r"’s", '', textunis)
    return textis
df.text = df.text.apply(lambda x: is_fix(x))



#punctuation

import re
def punc_fix(textpun):
    textunpun=re.sub(r'[^\w\s]', '', textpun)
    return textunpun
df.text = df.text.apply(lambda x: punc_fix(x))


# Replace all of the digits in the string with an empty string.
import re
def num_fix(textnum):
    textnonum=re.sub(r'[0-9]', '', textnum)
    return textnonum
df.text = df.text.apply(lambda x: num_fix(x))


# Python3 code to remove whitespace
def removewh(textwh):
    return " ".join(textwh.split())
df.text = df.text.apply(lambda x: removewh(x))


# Python3 code to lowercase
def lowercase(textU):
    return textU.lower()
df.text = df.text.apply(lambda x: lowercase(x))



#Removing Stop Words
import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
print(stopwords.words('english'))
from nltk.tokenize import word_tokenize

# Bring in the default English NLTK stop words
stoplist = stopwords.words('english')

# Open a file and read it into memory
file = open('Additional stop words.txt')
texti = file.read()

# Apply the stoplist to the text
additional_stopwords = [word for word in texti.split() if word not in stoplist]
stoplist += additional_stopwords

for i in range(len(stoplist)):
    stoplist[i] = stoplist[i].lower()

print (stoplist)

def token(textUnT):
    text_tokens = word_tokenize(textUnT)
    tokens_without_sw = [word for word in text_tokens if not word in stoplist]
    return tokens_without_sw
df.text = df.text.apply(lambda x: token(x))


#Lemmatization
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()

df.text = df.text.apply(lambda x: [lm.lemmatize(y) for y in x])
#Tokenize, remove stop words, and non-essential POS
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')  # for pos_tag function
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag  # for pos_tag function
print(stopwords.words('english'))
# Bring in the default English NLTK stop words
stoplist = stopwords.words('english')
# Open a file and read it into memory
file = open('Additional stop words.txt')
texti = file.read()
# Apply the stoplist to the text
additional_stopwords = [word for word in texti.split() if word not in stoplist]
stoplist += additional_stopwords
stoplist = [word.lower() for word in stoplist]
additional_stopwords = [word.lower() for word in additional_stopwords]
for i in range(len(stoplist)):
    stoplist[i] = stoplist[i].lower()
print(stoplist)
def token(textUnT):
    text_tokens = word_tokenize(textUnT)
    # Tag the parts of speech of the tokens
    pos_tags = pos_tag(text_tokens)
    # Keep only the words that are verbs, nouns, adjectives, or adverbs
    tokens = [word for word, tag in pos_tags if tag in ['NN','NNS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP']]
    # Remove stop words and return the filtered list of tokens
    tokens_without_sw = [word for word in tokens if not word in stoplist]
    return tokens_without_sw
filtered_df.text = filtered_df.text.apply(lambda x: token(x))



#wordCloud
from wordcloud import WordCloud
# Flatten the list of tokens into a single list
all_tokens = filtered_df['text'].sum()
# Join the list of tokens into a single string
all_text = ' '.join(all_tokens)
# Generate a wordcloud from the string
wordcloud = WordCloud(max_font_size=30, background_color='white').generate(all_text)
#, contour_color = 'black', contour_width = 2, color_func = lambda *args, **kwargs: 'black'
# Set the width and height of the figure
plt.figure(figsize=(80, 50))
# Display the wordcloud
plt.imshow(wordcloud, interpolation='bilinear')
# Turn off the axis labels
plt.axis("off")
# Set the title of the plot
plt.title("Word Cloud of Tweets")
# Display the plot
plt.show()

#get the top frequent words
import nltk
from nltk.probability import FreqDist
# Tokenize the text
tokens = nltk.word_tokenize(all_text)
# Count the frequency of each word
fdist = FreqDist(tokens)
# Convert the FreqDist object to a dataframe
import pandas as pd
word_freq_df = pd.DataFrame(fdist.items(), columns=['word', 'frequency'])
# Sort the dataframe by frequency
word_freq_df = word_freq_df.sort_values(by='frequency', ascending=False)





#LDA
import numpy as np
import json
import glob
#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
#spacy
import spacy
from nltk.corpus import stopwords
#vis
import pyLDAvis
import pyLDAvis.gensim_models
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
id2word = corpora.Dictionary(filtered_df.text)
corpus = []
for wordi in filtered_df.text:
    new = id2word.doc2bow(wordi)
    corpus.append(new)
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=50,
                                           passes=10,
                                           alpha="auto")
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
vis





#LSA
#LSA Topic Modeling Using sklearn
from sklearn.decomposition import TruncatedSVD
# Create an instance of the TruncatedSVD class
svd = TruncatedSVD(n_components=10, random_state=100)
corpus_matrix = gensim.matutils.corpus2dense(corpus, num_terms=len(id2word))
# Fit the model to the data
svd.fit(corpus_matrix)
# Get the topics
topics = svd.components_
num_top_words = 10
for topic_idx, topic in enumerate(topics):
    print("Topic {}:".format(topic_idx))
    top_words_idx = topic.argsort()[:-num_top_words-1:-1]
    top_words = [id2word.get(i) for i in top_words_idx]
    top_weights = topic[top_words_idx]
    for word, weight in zip(top_words, top_weights):
        print("{} - {}".format(word, weight))
    print("\n")






#LSA Topic Modeling Using gensim
from gensim.models.lsimodel import LsiModel
# Build the LSI model
lsi_model = LsiModel(corpus=corpus, num_topics=10, id2word=id2word)
# Print the topics
print(lsi_model.show_topics(num_topics=10, num_words=10))