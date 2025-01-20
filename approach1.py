import nltk
from collections import Counter
import heapq
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup

# Download necessary NLTK datasets (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Preprocess:
    def __init__(self):
        pass

    def toLower(self, x):
        '''Converts string to lowercase'''
        return x.lower()

    def sentenceTokenize(self, x):
        '''Tokenizes document into sentences'''
        sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        sentences = sent_tokenizer.tokenize(x)
        return sentences

    def preprocess_sentences(self, all_sentences):
        '''Tokenizes sentences into words, removes punctuations, stopwords and performs tokenization'''
        word_tokenizer = nltk.RegexpTokenizer(r"\w+")
        sentences = []
        special_characters = re.compile("[^A-Za-z0-9 ]")
        for s in all_sentences:
            # remove punctuation
            s = re.sub(special_characters, " ", s)
            # Word tokenize
            words = word_tokenizer.tokenize(s)
            # Remove Stopwords
            words = self.removeStopwords(words)
            # Perform lemmatization
            words = self.wordnet_lemmatize(words)
            sentences.append(words)
        return sentences

    def removeStopwords(self, sentence):
        '''Removes stopwords from a sentence'''
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in sentence if token not in stop_words]
        return tokens

    def wordnet_lemmatize(self, sentence):
        '''Lemmatizes tokens in a sentence'''
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token, pos='v') for token in sentence]
        return tokens

    def complete_preprocess(self, text):
        '''Performs complete preprocessing on document'''
        # Convert text to lowercase
        text_lower = self.toLower(text)
        # Sentence tokenize the document
        sentences = self.sentenceTokenize(text_lower)
        # Preprocess all sentences
        preprocessed_sentences = self.preprocess_sentences(sentences)
        return preprocessed_sentences

class NewsSummarization:
    def __init__(self):
        self.preprocessor = Preprocess()

    from bs4 import BeautifulSoup
import requests

class NewsSummarization:
    def __init__(self):
        self.preprocessor = Preprocess()

    def fetch_article(self, url):
        '''Fetches the article content from a URL'''
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract article text
            paragraphs = soup.find_all('p')
            article = " ".join([para.get_text() for para in paragraphs])

            # Clean up the article to remove unwanted sections like navigation, headers, etc.
            unwanted_tags = soup.find_all(['header', 'footer', 'nav', 'aside', 'script', 'style'])
            for tag in unwanted_tags:
                tag.decompose()  # Removes the tag from the soup object

            # After cleanup, we extract the content again
            cleaned_article = " ".join([para.get_text() for para in soup.find_all('p')])
            return cleaned_article
        except requests.exceptions.RequestException as e:
            print("Error fetching the article:", e)
            return None


    def extractive_summary(self, text, sentence_len=8, num_sentences=3):
        '''Generates extractive summary of num_sentences length using sentence scoring'''
        word_frequencies = {}
        # Preprocess and tokenize article
        tokenized_article = self.preprocessor.complete_preprocess(text)
        # Calculate word frequencies
        for sentence in tokenized_article:
            for word in sentence:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
        # Get maximum frequency for score normalization
        maximum_frequency = max(word_frequencies.values())
        # Normalize word frequency
        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / maximum_frequency)
        sentence_scores = {}

        # Score sentences by adding word scores
        sentence_list = nltk.sent_tokenize(text)
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) > sentence_len:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]
        # Get sentences with largest sentence scores
        summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        # Join and get extractive summary
        summary = ' '.join(summary_sentences)
        return summary

    def summarize_article(self, url, num_sentences=3):
        '''Fetches an article and returns its summary'''
        article = self.fetch_article(url)
        if article:
            summary = self.extractive_summary(article, num_sentences=num_sentences)
            return summary
        else:
            return "Could not fetch the article."

# Example usage
if __name__ == "__main__":
    url = input("Enter the article URL: ")
    summarizer = NewsSummarization()
    summary = summarizer.summarize_article(url, num_sentences=3)
    print("\nSummary:\n", summary)

