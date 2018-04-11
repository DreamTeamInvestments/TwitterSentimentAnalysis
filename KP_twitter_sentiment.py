# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:25:22 2017

@author: fgarcia
"""
import re
import json
import pandas as pd 
from textblob import TextBlob
import numpy as np
from dateutil import parser
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
from collections import Counter 
 
class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # Fetch tweets from input file. 
        tweets = self.get_tweets()
        
        # Collecting date of tweets. 
        datetweets = [tweet['date'] for tweet in tweets]
        date_data = [ w.replace('T', ' ') for w in datetweets ]
        date_data = [parser.parse(date) for date in date_data]
        
        # Collecting polarity of tweets. 
        polaritytweets = [tweet['polarity'] for tweet in tweets]
        
        # Specify length of moving average calculation. 
        n = 60
        polarity_data_pre = self.moving_average(polaritytweets, n)
        polarity_data = polarity_data_pre.tolist()
        
        # Create a time series plot with cumulative tweet data. 
        plt.plot(date_data[n-1:], polarity_data)
        plt.title('Polarity data for Kite Pharma search')
        plt.xlabel('Dates')
        plt.ylabel('Polarity Index')

        # picking positive tweets from tweets
        ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
        # percentage of positive tweets
        print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))
        # picking negative tweets from tweets
        ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
        # percentage of negative tweets
        print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))
        # percentage of neutral tweets
        print("Neutral tweets percentage: {} % ".format(100*(len(tweets) - len(ntweets) - len(ptweets))/len(tweets)))
     
        # printing first 5 positive tweets
        print("\n\nPositive tweets:")
        for tweet in ptweets[:50]:
            print(tweet['text'])
            
        # printing first 5 negative tweets
        print("\n\nNegative tweets:")
        for tweet in ntweets[:50]:
            print(tweet['text'])
 
    def moving_average(self, a, n):
        
        # Define the window for calculation of the moving average. 
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
    def get_tweet_polarity(self,tweet): 
        '''
        Utility function to estimate polarity of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        
        return analysis.sentiment.polarity 
        
    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
      
    def tokenize(self,s):
        
        emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [oO\-]? # Nose (optional)
            [D\)\]\(\]/\\OpP] # Mouth
        )"""
        
        regex_str = [
        emoticons_str,
        r'<[^>]+>', # HTML tags
        r'(?:@[\w_]+)', # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
     
        r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
        r'(?:[\w_]+)', # other words
        r'(?:\S)' # anything else
        ]
        tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
        
        return tokens_re.findall(s)
     
    def preprocess(self, s, lowercase=False):
        
        emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [oO\-]? # Nose (optional)
            [D\)\]\(\]/\\OpP] # Mouth
        )"""
                
        emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
        
        tokens = self.tokenize(s)
        if lowercase:
            tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
        return tokens
        
    def get_tweets(self):
        '''
        Main function to fetch tweets and parse them.
        ''' 
        # Create a directory to collect all Twitter data. 
        tweets_data_path = '/Users/fgarcia/Google Drive/Miscellaneous/Personal Documents/Investment Files/Twitter Sentiment Modeling/tweets3.json'
        tweets_file = open(tweets_data_path, "r")
        raw_data = tweets_file.read()
        
        # Format the data in order to be parsed by json reader. 
        tweaked_data = raw_data.replace('}, {', '},SPLIT {')
        split_data = tweaked_data.split(',SPLIT')
        parsed_data = [json.loads(bit_of_data) for bit_of_data in split_data]
        
        # Create a dataframe for the Twitter data.     
        tweets = pd.DataFrame()
        tweets['date'] = map(lambda tweet: tweet['timestamp'], parsed_data)
        tweets['text'] = map(lambda tweet: tweet['text'], parsed_data) 
        tweets['retweets'] = map(lambda tweet: tweet['retweets'], parsed_data) 
        
        # Find common terms. 
        #stopwords = nltk.download('stopwords')
        punctuation = list(string.punctuation)
        stop = stopwords.words('english') + punctuation + ['Kite','Pharma','KITE','com','\xa0','\u2026','twitter','Inc','rt', 'via']
        
        count_all = Counter()
        for i in range(len(tweets['text'])):
            
            tweets_iter = tweets['text'].iloc[i]
            terms_all = [ term for term in self.preprocess(tweets_iter) if term not in stop and not term.startswith(('#', '@'))] 
            
            # Update the counter
            count_all.update(terms_all)
        
        # Print the first 20 most frequent words
        print(count_all.most_common(50))
        
        # parsing tweets one by one
        tweet = []
        for i in range(len(tweets)):
            # empty dictionary to store required params of a tweet
            parsed_tweet = {}
 
            # saving date of the tweet. 
            parsed_tweet['date'] = tweets['date'].iloc[i] 
             
            # saving text of tweet
            parsed_tweet['text'] = tweets['text'].iloc[i]
            
            # saving sentiment of tweet
            parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweets['text'].iloc[i])
            
            # saving polarity of tweet
            parsed_tweet['polarity'] = self.get_tweet_polarity(tweets['text'].iloc[i])
 
            # appending parsed tweet to tweets list
#            if tweets['retweets'].iloc[i] > 0:
            # if tweet has retweets, ensure that it is appended only once
            if parsed_tweet not in tweet:
                tweet.append(parsed_tweet)
            else:
                continue
                    
        return tweet
 
def main():
    # creating object of TwitterClient Class
    TwitterClient()
    
if __name__ == "__main__":
    # calling main function
    main()