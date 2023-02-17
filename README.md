# Sentiment Analysis on collected Tweets to Predict Real State Pricing Trend in 2010 to 2021

## Author: Milad(Mylo) Ebtedaei

This work was done for a private project and the dataset cannot be released because I don't own the copyright. However, everything in this repository can be easily modified to work with other datasets.

## Dataset Information
We use and compare various different methods for sentiment analysis on tweets collected from Twitter with tweepy tweepy which is a python library for accessing the Twitter API. The dataset is a combined csv file. 

## Requirements
There are some general library requirements as follows.

- numpy
- scikit-learn
- scipy
- nltk
- spacy
- matplotlib
- gensim
- panda
- vader sentiment

## What is Latent Dirichlet Allocation?
A common topic modeling method is Latent Dirichlet Allocation first proposed by David Blei, Andrew Ng und Michael I. Jordan in 2003. It is a general statistical model that allows to find and identify topics within documents. It posits that each document consists of a mixture of topics, and each word in a document is attributed to a specific topic. The sets of words that make up the topics are then in an iterative training process identified. The only variable that has to be specified beforehand is the number of topics.

## Preprocessing
In order to achieve good results we need to preprocess the collected tweets. Ideally, we want to end up with words that are decisive for a topic and also with as few words as necessary. In a first step we will do some text cleaning on tweets such as removing any type of punctuation, quotation marks, URL's, stopword and etc. Then we will filter out all words that are not nouns, adjectives, adverbs or verbs. Further, we will lemmatize the remaining words, a process by which inflected words are transformed into their base form. These steps greatly decrease the size of unique words. 
In a final preprocessing step we will calculate bag-of-words representations for each document, which could be understood as a vector pointing into a specific direction depending on which words and how often they were used in a document.

## Word Clouds
Plotting word clouds can also give us a good idea about which topics have been recognized. A word cloud shows common words used in all the titles.
![Screenshot 2023-02-11 123848](https://user-images.githubusercontent.com/121390440/218272732-623d009d-7194-4b60-a35c-cd8721e393c4.png)

## pyLDAvis
Another popular visualization method is pyLDAvis:
![Screenshot 2023-02-11 100150](https://user-images.githubusercontent.com/121390440/218265654-28030bbd-d9a7-453e-8f5c-e0cb97f1e957.png)

## Interesting facts from exploratory data analysis
Less 0.01% users will push tweets with their locations.
More than 65.6% users will write the locations in their profile, although very few of them don't live on Earth according to that fact. In this project we have used the location that is mentioned in the profile.

## Original Development
- Extract Twitter Data, preprocess data in Python
- Combine collected tweets in the form of dataframes together 
- Perform exploratory data analysis and text clening on tweets
- Connect with Plotly for interactive dashboard 

## Challenges
Unstructured tweet texts may contain messy code and emoji characters
Most of the tweets were created by bots or were advertisement
It's hard to get Tweiter's approval to collect the tweets with tweepy for more than 500.000 tweets
Plotly doesn't have well-document on reference making customize dashboard much harder
 
## Questions?
Email the author at mylo.ebted@gmail.com
