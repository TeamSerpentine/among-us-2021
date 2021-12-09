import json
from collections import Counter
import math
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.stem import WordNetLemmatizer
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd


def preprocessing():
    @dataclass
    class ChatLog:
        colorName: str
        imposter: bool
        name: str
        message: str
        wordlist: list
        TFIDF: float

    # nltk.download('punkt')
    data_list = []
    folder_directory = "CrewmateOCR/"
    json_files = os.listdir(folder_directory)
    total_words_crew = 0  # the total number of words in the crew chat messages
    total_words_imposter = 0
    words_list = []
    unique_words_list = []  # number of unique words, for laplace smoothing
    skipped_json = 0  # amount of JSON skipped due to incomplete imposter data
    processed_json = 0  # amount of JSON used for training data
    all_messages = []

    for jsonLocation in json_files:

        # Check and open .txt file and store imposter color
        if not ".json" in jsonLocation:  # only look at the json files in the folder
            continue
        json_location_file = folder_directory + jsonLocation
        txt_file_location = json_location_file.replace(".json", ".txt")  # grab the imposter colors
        if not os.path.isfile(txt_file_location):
            skipped_json = skipped_json + 1
            continue
        impostor_file = open(txt_file_location, 'r')
        impostor_colors = []
        lines = impostor_file.readlines()
        number_of_imposters = int(lines[0])
        if number_of_imposters != (
                len(lines) - 1):  # check if the #detected imposters is equal to the total amount of imposters
            skipped_json = skipped_json + 1
            continue
        processed_json = processed_json + 1
        for line in lines[1:]:
            impostor_colors.append(line)


        # Open JSON file with messages
        json_file = open(json_location_file, 'r').read()
        ocr_info = json.loads(json_file, strict=False)
        for chat in ocr_info["ChatMessages"]:

            # Add labels to messages
            chatMessage = chat["MESSAGE"]
            is_imposter = False
            for color in impostor_colors:
                if chat["COLORNAME"] == color:
                    is_imposter = True

            # Tokenize, stem, and filter messages
            tokenized_word = word_tokenize(chat["MESSAGE"])
            tokens = []
            # stemmer = SnowballStemmer("english") # intialize stemmer
            lemmatizer = WordNetLemmatizer()  # initialize lemmatizer
            stop_words = set(nltk.corpus.stopwords.words('english'))

            for word in tokenized_word:
                # word = stemmer.stem(word)
                word = lemmatizer.lemmatize(word)
                word = word.lower()  # make all messages lowercase
                word = "".join(char for char in word if char.isalpha())  # Filter all non-alphabet characters
                if len(word) > 1 and word not in stop_words:
                    tokens.append(word)
                else:
                    continue
            all_messages.append(tokens)
            newLog = ChatLog(chat["COLORNAME"], is_imposter, chat["NAME"], chat["MESSAGE"], tokens, 0)
            data_list.append(newLog)

            # Create list with vocab of imposter and crew
            if is_imposter:
                total_words_imposter = total_words_imposter + len(tokens)
            else:
                total_words_crew = total_words_crew + len(tokens)

            # Create list of unique words
            for token in tokens:
                words_list.append(token)
                if token not in unique_words_list:
                    unique_words_list.append(token)

    # Find frequency of words           
    word_freq = Counter()
    for word in words_list:
        word_freq[word] += 1

    # Create vocabulary 
    vocab = [word for word in word_freq]
    df = pd.DataFrame(0, index=np.arange(len(data_list)), columns=vocab)

    # TF-IDF
    for i in range(len(data_list)):
        # count words in every message
        words = data_list[i]
        c = Counter(words.wordlist)
        for word in words.wordlist:
            # calculate TF
            freq_term = int(('%d' % (c[word])))
            num_word = len(words.wordlist)
            tf = (freq_term / num_word)
            # calculate IDF
            documents_w_t = word_freq[word]  # = denominator of IDF formula
            idf = math.log2(len(data_list) / documents_w_t)

            # calculate TF-IDF (TF * IDF)
            tfidf = tf * idf
            df[word].iloc[i] = tfidf

    # Create list of labels
    labels = []
    for data in data_list:
        if not data.imposter:
            labels.append(0)
        else:
            labels.append(1)

    # Convert list to array and rename data
    labels = np.array(labels)
    X = df
    y = labels

    # Drop all empty rows
    X, y = pd.DataFrame(X), pd.DataFrame(y)
    indices = X.index[X.eq(0).all(1)]
    X.drop(indices, inplace=True)
    y.drop(indices, inplace=True)
    y = y.values.ravel()
    X = np.array(X)

    return X, y
