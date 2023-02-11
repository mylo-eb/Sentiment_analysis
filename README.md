# Sentiment Analysis on collected Tweets to Predict Real State Pricing Trend in 2010 to 2021
![Screenshot 2023-02-11 100150](https://user-images.githubusercontent.com/121390440/218265654-28030bbd-d9a7-453e-8f5c-e0cb97f1e957.png)

## Author: Milad(Mylo) Ebtedaei

This work was done for a private project and the dataset cannot be released because I don't own the copyright. However, everything in this repository can be easily modified to work with other datasets.

## Dataset Information
We use and compare various different methods for sentiment analysis on tweets collected from Twitter with tweepy tweepy which is a python library for accessing the Twitter API. The dataset is a combined csv file. 

## Requirements
There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.

- numpy
- scikit-learn
- scipy
- nltk


The library requirements specific to some methods are:
.
.
.
Analyzes sentiment from Twitter tweets on housing prices to determine impact of public sentiment on Real Estate pricing.
This repository contains all the associated work that has been done for the area which includes:
Tweepy as a Python library for accessing the Twitter API and collecting tweets
Notebooks associated with data engineering, LDA and regression Modeling
Web App

## Interesting facts from exploratory data analysis
Less 0.01% users will push tweets with their locations.
More than 65.6% users will write the locations in their profile, although very few of them don't live on Earth according to that fact. In this project we have used the location that is mentioned in the profile.

## Orignal Development
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
