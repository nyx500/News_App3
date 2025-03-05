# lime_functions.py

import os
# Imports basic data processing libs
import pandas as pd
import numpy as np
import re
# For app creation
import streamlit as st
# Imports text processing libs + downloads required word packages
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("punkt") # For tokenizer
nltk.download("punkt_tab")
nltk.download("stopwords") # Download stopwords list
# Creates the global set of stopwords (once, for efficiency, not to have to do it for every individual news prediction)
stop_words = set(stopwords.words('english'))
# For extracting named entity features and part-of-speech syntactic features
import spacy
# For emotion feature processing
from nrclex import NRCLex
# Textstat library for extracting readability features for inputted news text
import textstat
# For Streamlit data visualization + charts
# Reference: https://docs.streamlit.io/develop/api-reference/charts/st.altair_chart
import altair as alt
# LIME explanation library for text-based explanations
from lime.lime_text import LimeTextExplainer


class BasicFeatureExtractor:
    """
        A class containing methods for extracting key discriminative features for 
        helping an ML-based classifier categorize real and fake news. Inc. lexical features such as
        normalized (by text length in word tokens) exclamation point count, as well as semantic
        features (e.g. pos emotion score)
    """
    
    def __init__(self):
        # Loads in the SpaCy model for POS-tag + NER extraction
        self.nlp = spacy.load("spacy_model")


    def extractExclamationPointFreqs(self, text):
        """
        Extracts the frequencies of exclamation points from a single news text. 
        
            Input Parameters:
                text (str): the news text to extract exclamation point frequencies from
    
            Output:
                excl_point_freq (float): the normalized exclamation point frequency for the text.
                Normalized by num of word tokens to handle varying text length datasets
        """
        # Counts the number of exclamation points in the text
        exclamation_count = text.count('!')
        # Tokenizes text for calculating text length
        word_tokens = word_tokenize(text)
        # Calculates the text length in number of word tokens
        text_length = len(word_tokens)
        # Normalizes the exclamation point frequency by text length in tokens
        return exclamation_count / text_length if text_length > 0 else 0 # Handles division-by-zero errs
    


    def extractThirdPersonPronounFreqs(self, text):
        """
        Extracts the normalized frequency counts of third-person pronouns in the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract pronoun features from.
            
            Output:
                float: Normalized third-person pronoun frequency.
        """
        # Creates a alphab-ordered list of English 3rd person pronouns
        third_person_pronouns = [
            "he","he'd", "he's", "him", "his",
            "her", "hers", 
            "it", "it's", "its",
            "one", "one's", 
            "their", "theirs", "them","they", "they'd", "they'll", "they're", "they've",   
            "she", "she'd", "she's"
        ]

        # Tokenizes text for calculating text length
        word_tokens = word_tokenize(text)

        # Gets the text length in num tokens
        text_length = len(word_tokens)

        # Counts the frequency of third-person pronouns in the news text; lowercases text to match the list of third-person pronouns above
        third_person_count = sum(1 for token in word_tokens if token.lower() in third_person_pronouns)

        # Normalizes the frequency by text length in word tokens
        return third_person_count / text_length if text_length > 0 else 0



    def extractNounToVerbRatios(self, text):
        """
        Calculates the ratio of all types of nouns to all types of verbs in the text
        using the Penn Treebank POS Tagset and the SpaCy library with the downloaded
        "en_core_web_lg" model.
        
            Input Parameters:
                text (str): the news text to extract noun-verb ratio features from.
            
            Output:
                float: Noun-to-verb ratio, or 0.0 if there are 0 verbs in text
        """
        
        # Converts the text to an NLP doc object using SpaCy
        doc = self.nlp(text)
        
        # Defines the Penn Treebank POS tag categories for nouns and verbs
        # Reference: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        verb_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        
        # Counts the freqs of both nouns and verbs based on the above Penn Treebank tags
        noun_count = sum(1 for token in doc if token.tag_ in noun_tags)
        verb_count = sum(1 for token in doc if token.tag_ in verb_tags)
        
        # Calculates and returns the noun-to-verb ratio (should be higher for fake news, as it had more nouns in EDA)
        return noun_count / verb_count if verb_count > 0 else 0.0 # Avoid division-by-zero error


    def extractCARDINALNamedEntityFreqs(self, text):
        """
        Extracts the normalized frequency of CARDINAL named entities in the text
        using the SpaCy library.
        
            Input Parameters:
                text (str): The text to extract the CARDINAL named entity frequencies from.
            
            Output:
                float: Normalized frequency (by number of tokens in the text) of CARDINAL named entities.
        """

        # Processes the text again with SpaCy to get NLP doc object
        doc = self.nlp(text)

         # Counts how many named entities have the label "CARDINAL"
        cardinal_entity_count = sum(1 for entity in doc.ents if entity.label_ == "CARDINAL")

        # Tokenizes the text
        word_tokens = [token for token in doc]
        
        # Count num of word toks
        text_length = len(word_tokens)

        # Returns the normalized frequency of CARDIAL named entities by word tok length
        return cardinal_entity_count / text_length if text_length > 0 else 0.0 # Avoid division-by-zero error


    def extractPERSONNamedEntityFreqs(self, text):
        """
        Extracts the normalized frequency of PERSON named entities in the text
        using the SpaCy library.
        
            Input Parameters:
                text (str): The text to extract the PERSON named entity frequencies from.
            
            Output:
                float: Normalized frequency (by number of tokens in the text) of PERSON named entities.
        """
        # Processes the text with SpaCy to get NLP doc object
        doc = self.nlp(text)
        
        # Counts how many named entities have the label "PERSON"
        person_entity_count = sum(1 for entity in doc.ents if entity.label_ == "PERSON")
        
        # Tokenizes the text
        word_tokens = [token for token in doc]
        
        # Counts num of word tokens
        text_length = len(word_tokens)
        
        # Returns the normalized frequency of PERSON named entities, normalized by dividing by text length in word tokens
        return person_entity_count / text_length if text_length > 0 else 0.0 # Avoids division-by-zero error


    def extractPositiveNRCLexiconEmotionScore(self, text):
        """
        Extracts the POSITIVE emotion score using the NRC Lexicon from the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract POSITIVE emotion score from.
            
            Output:
                float: the POSITIVE words NRC Lexiconemotion score.
        """
        # Converts the text to lowercase to find (uncased) words in the lexicon
        text = text.lower()

        # Creates an NRC Emotion Lexicon object to extract emotion word freqs from
        emotion_obj = NRCLex(text)
        
        # Return the POSITIVE emotion score (use "get" to default to 0.0 if for some reason it is not found)
        return emotion_obj.affect_frequencies.get("positive", 0.0) 


    def extractTrustNRCLexiconEmotionScore(self, text):
        """
        Extracts the TRUST emotion score using the NRC Lexicon from the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract TRUST emotion score from.
            
            Output:
                float: the extracted TRUST emotion NRC lexicon score.
        """
        # Converts the text to lowercase to find (uncased) words in the lexicon
        text = text.lower()

        # Creates an NRC Emotion Lexicon object to extract emotion word freqs from
        emotion_obj = NRCLex(text)
        
        # Returns the TRUST emotion score (use "get" to default to 0.0 if for some reason it is not found)
        return emotion_obj.affect_frequencies.get("trust", 0.0)


    def extractFleschKincaidGradeLevel(self, text):
        """
        Extracts the Flesch-Kincaid Grade Level score for the input text.
        
        Input Parameters:
            text (str): the news text to calculate the Flesch-Kincaid Grade Level score.
        
        Output:
            float: the Flesch-Kincaid Grade Level score for the text.
        """
        return textstat.flesch_kincaid_grade(text)
        
    def extractDifficultWordsScore(self, text):
        """
        Extracts the number of difficult words (not in the Dall-Chall word list) in the input text using the textstat library.
        Reference about the Dall-Chall word list used to compute these scores: 
            - https://readabilityformulas.com/word-lists/the-dale-chall-word-list-for-readability-formulas/
        
        Input Parameters:
            text (str): the news text to calculate the difficult words score.
        
        Output:
            float: the number of difficult words score for the text.
        """
        # Converts text to lowercase to match if words are in difficult words list
        text = text.lower()
        
        # Returns difficult words textstat score; the higher the score, the more non-Dall Chall words, the more coplex the text
        return textstat.difficult_words(text)
    

    def extractCapitalLetterFreqs(self, text):
        """
        Extracts the normalized frequency of capital letters in the input text.
        Normalized by the total number of word tokens.
    
            Input Parameters:
                text (str): The news text to extract capital letter frequencies from.
            
            Output:
                float: Normalized frequency of capital letters in the text.
        """
        # Counts the number of capital letters in the text
        capital_count = sum(1 for char in text if char.isupper())
        
        # Tokenizes the text
        word_tokens = word_tokenize(text)

        # Counts the number of word tokens in the text
        text_length = len(word_tokens)
        
        # Normalizes the frequency of capital letters
        return capital_count / text_length if text_length > 0 else 0.0 # Avoids division-by-zero error
    

    def extractFeaturesForSingleText(self, text):
        """
        Extracts the basic features for classifying an individual news text, to concatenate with the fastText embeddings.

            - It outputs a single-row DataFrame because this will be used to extract the features
            for each perturbation of the text created to test the impact of removing different features on final probabilities. 
            - Each single-row DataFrame for the perturbed texts will then be concatenated into a multi-row DataFrame, storing the
            features for all of the perturbed texts in a single table, to use as model inputs.
    
        Input Parameters:
            text (str): The text to extract features from.
    
        Output:
            pd.DataFrame: A single-row DataFrame storing the extracted basic feature values for this text, each feature per column
        """
        # Extracts the features and stores them in a dict
        feature_dict = {
            "exclamation_point_frequency": self.extractExclamationPointFreqs(text),
            "third_person_pronoun_frequency": self.extractThirdPersonPronounFreqs(text),
            "noun_to_verb_ratio": self.extractNounToVerbRatios(text),
            "cardinal_named_entity_frequency": self.extractCARDINALNamedEntityFreqs(text),
            "person_named_entity_frequency": self.extractPERSONNamedEntityFreqs(text),
            "nrc_positive_emotion_score": self.extractPositiveNRCLexiconEmotionScore(text),
            "nrc_trust_emotion_score": self.extractTrustNRCLexiconEmotionScore(text),
            "flesch_kincaid_readability_score": self.extractFleschKincaidGradeLevel(text),
            "difficult_words_readability_score": self.extractDifficultWordsScore(text),
            "capital_letter_frequency": self.extractCapitalLetterFreqs(text),
        }

        # Converts the dict above to a DataFrame (it will contain a single row, as this is just one text)
        feature_df = pd.DataFrame([feature_dict])

        # Returns the dict with a single row, columns containing the feature names
        return feature_df



def preprocessTextForFastTextEmbeddings(text):
    """
        - Basic text preprocessing function required before applying the fastText dense embedding model. 
        - It only cleans text from extra whitespace and newlines, as well as lowercasing,
        in preparation for training a fastText model, as this kind of model cannot handle newlines and trailing spaces.
       -  Does not remove punctuation as based on feature analysis, this can be an important distinguishing factor for fake news,
       as shown before in the EDA.
       - Does not remove stopwords because they could be important to fastText model, which has been autotuned to poss. include
       n-grams (bi/trigrams), where stopwords can be important to determining meaning of phrasal verb patterns, collocations etc.
        - Also does not aggressively apply normalization (lemmatize/stemming) words as this is handled by the fastText sub-word model.
        
        Input Parameters:
            text (str): the text to preprocess
        Output:
            text (str): the cleaned text
    """
    # Converts the text to lowercase
    text = text.lower()
   
    # Removes the newlines and extra whitespace (the \s regex matches any whitespace character including tabs/newlines)
    text = re.sub(r"\s+", ' ', text)

    # Strips the remaining trailing whitespace
    text = text.strip()
   
    return text


def getFastTextEmbedding(model, text):
    """
    Extracts the fastText dense embedding for a text using a pre-trained fastText model.
    
    Input Parameters:
        model: the pre-trained fastText model, trained on all-four combined text datasets
        text (str): the text to extract the embedding for
        
    Output:
        numpy.ndarray: the fastText dense embedding as a numpy array
    """
    # Cleans the text using the above helper function
    text = preprocessTextForFastTextEmbeddings(text)

    # Extracts the dense embedding for this text from the fastText model
    return model.get_sentence_vector(text)


def combineFeaturesForSingleTextDF(single_text_df, scaler, feature_cols):
    """
        Scales and combines extra engineered engineered features and concatenates them into a row with the fastText
        embeddings, outputting a 2D numpy array with a single row containing the concatenated features.

        Input Parameters:

        single_text_df (pd.DataFrame): a pandas DataFrame containing a single row with a text, fastText embedding, and extra feature scores
        scaler (sklearn.preprocessing.StandardScaler): a pre-fitted, saved StandardScaler instance for scaling the engineered features
        feature_cols (list): the list of column names for extracting the engineered features
        
        Output:
            final_vector (numpy.ndarray): the final combined embeddings + feat. vector for inputting into the classifier model
    """

    # Extracts the fastText embedding from the single-text DataFrame (will be at row at index 0, as there is only 1 row here) 
    embedding = single_text_df["fasttext_embeddings"].iloc[0]

    # Checks if it a np array type, otherwise if was converted into string instead, extract the values in between the
    # square brackets, use space as a separate to identify each float, and convert to np array using .fromstring
    if not isinstance(embedding, np.ndarray):
        embedding = np.fromstring(embedding.strip("[]"), sep=" ")
        
    # Scales the extra features using the pre-fitted StandardScaler
    engineered_features = scaler.transform(single_text_df[feature_cols])

    # Get max val from engineered features to check if it is a val greater than 1, to see if scaled properly
    max_abs_value_features = np.max(np.abs(engineered_features))
    
    # If max feature value is greater than 1, then scale it down by a scaling factor
    if max_abs_value_features > 1:
        # Calculates scaling factor to get all features under 1: 1 divided by value of max feature
        scale_factor = 1 / max_abs_value_features
        # Applies the scale-factor element-wise to the engineered features, to make them all less than absolute value of 1
        engineered_features = engineered_features * scale_factor

    # Adds a new dimension to the embedding array to make it 2D for model training
    embedding = embedding.reshape(1, -1) 

    # Concatenates the single-row containing the embedding with the engineered features for the final text
    final_vector = np.hstack([embedding, engineered_features])

    # Return the embedding + scaled features vector for model predictions
    return final_vector


def explainPredictionWithLIME(
    fasttext_model,
    classifier_model,
    fitted_scaler,
    text,
    feature_extractor,
    num_features=50,
    num_perturbed_samples=500
):
    """
        - Extracts features from a user-inputted news text, predicts its probability of being fake news by using a custom pipeline
        that first extracts FastText embeddings, extracts and scales the extra features, and inserts them into
        a pre-trained Passive-Aggressive classifier, wrapped in a Calibrated Classifier, for outputting fake vs real news probabilities.
        - It uses LIME to generate local explanations (word and feature imporances) for the individual prediction.

        Input Parameters:

        fasttext_model (<fasttext.FastText._FastText> model): a supervised, pre-trained FastText model for extracting 
                                                                  dense text embeddings, trained on training data from
                                                                  four domain-specific datasets
                                                    
        classifier_model (sklearn.calibration.CalibratedClassifierCV): the pre-trained Calibrated Classifier that is
                                                                    wrapping a Passive-Aggressive base classifier for 
                                                                    outputting probabilities

        fitted_scaler (sklearn.preprocessing._data.StandardScaler): a pre-trained scikit-learn StandardScaler, on the same 
                                                                 combined dataset, for scaling the extra features

        text (str): the text to get the fake or real news prediction for

        feature_extractor (BasicFeatureExtractor): a class instance for extracting extra semantic and linguistic features from a text

        num_features (int): the number of top word features LIME should output importance scores for

        num_perturbed_samples (int): number of perturbed samples to use for LIME explanations

        Output:
            dict: stores a summary of the information for explaining the prediction using LIME. Contains

                - "explanation_object": the default LIME explanation object returned by the LIME text explainer
                - "word_features_list": the list of tuples containing words and their LIME importance scores
                - "extra_features_list": a list of tuples containing the extra engineered features and their importance scores
                - "highlighted_text": the original text string formatted with HTML markup tags for displaying color-coded word features
                - "probabilities": an array of probabilities for [real, fake] news
                - "main_predicition": the final prediction, meaning integer of 0 for real news, 1 for fake news
    """

    # Instantiates a Lime Text Explainer
    text_explainer = LimeTextExplainer(class_names=["real", "fake"]) # Requires a mapping for 0 = real, 1 =fake

    # Extracts the extra engineered semantic and linguistic features for this text into a single-row Dataframe
    single_text_df = feature_extractor.extractFeaturesForSingleText(text)

    # Adds the original text as a column to the single-line features DF
    single_text_df["text"] = text

    # Extracts and stores ONLY the extra features (without the original text) in a features only Dataframe
    extra_features = single_text_df.drop("text", axis=1)

    # Extracts (as a list) the extra feature column names from that single-row features DataFrame for later mappings
    extra_feature_names = extra_features.columns.tolist()

    # Extracts the fastText embedding for the user-inputted text, inc. preprocessing function
    fasttext_embedding = getFastTextEmbedding(fasttext_model, text)

    # Stores the np fastText embedding in the single-row Dataframe
    single_text_df["fasttext_embeddings"] = [fasttext_embedding]

    # Returns the scaled and combined feature vector for the original input text
    text_features = combineFeaturesForSingleTextDF(single_text_df, fitted_scaler, extra_feature_names)

    # Predicts the news class and the class probabilities using the pre-trained Calib wrapping Passive-Aggressive news classifier
    text_prediction = classifier_model.predict(text_features)[0] # Extracts the single pred., as classifier returns a 2D array

    # Extracts the probability array from the CalibratedClassifierCV's predict_proba function
    text_probability_array = classifier_model.predict_proba(text_features)[0] 

    
    def predict_fn(perturbed_texts):
        """
        An inner function which LIME uses in order to predict probabilities of randomly-changed, perturbed texts.
        NB: use an inner func. as this is only to be integrated with the LIME process, and has to access
        many variables inside the explainPredictionWithLIME, such as classifier models, scalers etc. to generate predictions.
        
        Input Parameters:
            perturbed_texts (list): a list of randomly changed, perturbed texts generated by LIME, with random words removed
        
        Output:
            perturbed_probs (numpy.ndarray): the outputted array of probabilities for the inputted perturbed news texts
        """

        # Stores the perturbed texts' extra features here
        perturbed_text_features_df_list = []
        
        # Iterates over each of the perturbed texts
        for perturbed_text in perturbed_texts:
            # Extracts the features for each perturbation of original text (outputs single row DF for later concat)
            df = feature_extractor.extractFeaturesForSingleText(perturbed_text)
            # Adds the actual text to the single-row DataFrame for this perturbed text
            df["text"] = perturbed_text
            # Adds the fastText embedding to the single-row DataFrame
            df["fasttext_embeddings"] = [getFastTextEmbedding(fasttext_model, perturbed_text)]
            # Ads the single-row DataFrame to the whole list of DataFrames for later concatenation
            perturbed_text_features_df_list.append(df)
        
        # Concatenates the single perturbed rows into one whole DataFrame
        perturbations_df = pd.concat(perturbed_text_features_df_list, axis=0, ignore_index=True)
        
        # Extracts, scales and concatenates extra features with fastText embeddings into arrays for model inputs
        perturbed_feature_arrays = []

        # Iterates over rows in the combined, perturbed features and text DataFrame for the changed samples
        for i in range(len(perturbations_df)):
            # Extracts a single row of perturbed features and text
            perturbed_text_row = perturbations_df.iloc[[i]]
            # Combines and scales the extra features with the fastText embeddings, outputs a single array of features
            perturbed_features = combineFeaturesForSingleTextDF(perturbed_text_row, fitted_scaler, extra_feature_names)
            # Appends the single text's features to the list storing all perturbed features for later stacking
            perturbed_feature_arrays.append(perturbed_features)
        
        # Stacks the list of rows of features into a single numpy features matrix (2D array)
        perturbed_feature_arrays = np.vstack(perturbed_feature_arrays)
        
        # Uses the trained model to output the probabilities for all perturbed samples
        perturbed_probs = classifier_model.predict_proba(perturbed_feature_arrays)
        
        # Return probs of permuted samples to LIME explainer to generate explanations based on prob differences to original text
        return perturbed_probs

    # Generates the LIME explanation using .explain_instance()
    # Reference: https://lime-ml.readthedocs.io/en/latest/lime.html
    """"
        Docs: "First, we generate neighborhood data by randomly perturbing features from the instance [...]
        We then learn locally weighted linear models on this neighborhood data to explain each of the classes in an interpretable way (see lime_base.py)."
    """
    explanation = text_explainer.explain_instance(
        text, # Original text
        # Docs: a required "classifier prediction probability function, which takes a numpy array and outputs prediction
        #  probabilities. For ScikitClassifiers, this is classifier.predict_proba."" This is the inner func. above.
        predict_fn,
        num_features=num_features, # Num of top/most important words to output LIME importance scores for
        num_samples=num_perturbed_samples, # Num of perturbed versions of the text to generate; the more the greater accuracy, but takes more time
        labels=[text_prediction] # "Explains" the main original text prediction, so words pushing to main prediction get POSITIVE scores
    )
    
    # Returns the word feature explanations as a list of tuples of (word, importance_score)
    word_features = explanation.as_list(label=text_prediction)

    # Filters out important words which are stopwords for more meaningful explanations
    word_features_filtered = [(word_feature[0], word_feature[1]) for word_feature in word_features
                             if word_feature[0].lower() not in stop_words] 

    # Sorts the text_feature_list in descending order by absolute value of importance scores
    word_feature_list_sorted = sorted(word_features_filtered, key = lambda x: abs(x[1]), reverse=True)

    print(f"\n\n{word_feature_list_sorted}\n\n")

    # Creates a list to store the extra features'importance in this list of tuples (feature_name, feature_importance)
    extra_feature_importances= []

    # Iterates over extra features
    for feature in extra_feature_names:

        # Creates a perturbed version of the original single text's row, but with the feature's value zeroed out to eval its importance
        perturbed_df = single_text_df.copy()

        # Zero out the current feature
        perturbed_df[feature] = 0

        # Outputs the features array for the original text but with this feature zeroed out
        features_perturbed = combineFeaturesForSingleTextDF(perturbed_df, fitted_scaler, extra_feature_names)

        # Returns the [real, fake] probability array for the text with the feature zeroed out
        perturbed_probability_array = classifier_model.predict_proba(features_perturbed)[0]
        
        # Calculate the importance of the current feature to the main prediction:
        # This is done by calculating the difference between the main pred probability for original and perturbed text
        feature_importance = text_probability_array[text_prediction] - perturbed_probability_array[text_prediction]

        # Append a tuple storing name of the zeroed-out current feature and its importance based on probability difference
        extra_feature_importances.append((feature, feature_importance))


    # Sorts the extra features scores by absolute importance in desc order
    extra_feature_importances.sort(key=lambda x: abs(x[1]), reverse=True)

    # Generates highlighted text using the func below --> it outputs a string with HTML-formatted text
    highlighted_text = highlightText(text, word_feature_list_sorted, text_prediction)
    
    # Returns the explanation dictionary with LIME explanations, sorted word + extra feature lists, highlighted text + main preds
    return {
        "explanation_object": explanation,
        "word_features_list": word_feature_list_sorted,
        "extra_features_list": extra_feature_importances,
        "highlighted_text": highlighted_text,
        "probabilities": text_probability_array,
        "main_prediction": text_prediction
    }




def highlightText(text, word_feature_list, text_prediction):
    """
    A function for highlighting the words in the input text based on their importance scores and class label the words are pushing
    the classifier towards using HTML <span> + inline CSS tags for color-coding for red=fake news, blue=real news.

    Input Parameters:
        text (str): the text to highlight
        word_feature_list (list of tuples): list of word-feature, importance-score tuples outputted by the LIME text explainer

    Output:
        str: HTML formatted string with tags designating the highlighted text
    """
    
    # Stores dicts containing pos, colors and opacities for highlighting parts of the text with HTML tags based on word importance
    highlight_positions = []
    
    # Calculates the maximum absolute importance score from all the word features
    max_importance = max(abs(importance) for feature, importance in word_feature_list)

    # Iterate over important words to hihglight them and find their pos in the text
    for word_feature, importance in word_feature_list:
        
        # Placeholder for where to start searching for in the original text that will be updated during the loop
        pos = 0 # Starts at first character in the text

        # Breaks out of loop when no more of this word feature are found in the text
        while True:

            # Finds occurence of important word in the remaining text
            pos = text.lower().find(word_feature.lower(), pos)
            
            # If word not found in remaining text, break out of while loop
            if pos == -1:
                break
                
            # Checks if word feature found is a valid word, not a sub-word of a different word
            # Returns its pos in the text (character indices) if it is a real word, None if it is not
            boundary_positions = detectWordBoundaries(text, pos, word_feature)

            # If word feature is just a subword, increment pos and move to next part of the text string, restart at beginning of while loop
            if boundary_positions is None:
                pos += 1 
                continue
            
            # If word features is a whole word, unpack its beginning and end pos (character indices)
            word_start_pos, word_end_pos = boundary_positions

            # Maps colors to which class the feature is pushing the classifier towards
            # If the word feature importance is > 0, it means the feature is pushing it towards the main prediction
            if importance > 0:
                # If feature has same importance score as the main prediction, make sure real news = blue, and fake = red
                color = "blue" if text_prediction == 0 else "red" 
            else:
                # If feature has same importance score as the main prediction, make sure if main pred is real news,
                # then opposing word feature will be in red, if main pred is fake, opposing feature will be blue
                color = "red" if text_prediction == 0 else "blue"
            
            # Adds the dict storing word positions, color, opacity for highlighting the whole text
            highlight_positions.append({
                "start": word_start_pos, 
                "end": word_end_pos, 
                "color": color,
                "opacity": abs(importance) / max_importance if max_importance != 0 else 0, # Maps color alpha channel to abs feat importance
                "text": text[word_start_pos:word_end_pos]
            })
            
            # Moves past this word by incrementing the string index to be at the character index just after this word
            pos = word_end_pos  

    # Sorts the word position dictionaries in ascending order based on word start index, to go from beginning to end of text
    highlight_positions.sort(key = lambda x: x["start"])
    
    # Merges the adjacent highlights of the same color to represent bigrams and trigrams and continuous text sequences, if have same color
    merged_positions = []

    # Only if there is more than 1 dictionary in the highlight_positions list then proceed
    if highlight_positions:

        # Extracts the first dictionary (storing text segment information) for highlighting
        current = highlight_positions[0]

        # Iterates through all of the next highlighting information dictionaries after this current one
        for next_pos in highlight_positions[1:]:

            # Checks if the next segment to be highlighted either starts right after current one ends or overlaps (add 1 to account for spaces!!!)
            if (next_pos["start"] <= current["end"] + 1 and
                # Only proceeds if the colors of highlighted words are the same
                next_pos["color"] == current["color"]):

                # Merges the words by extending current end position to be the one LATER at the end of the next segment instead
                current["end"] = max(current["end"], next_pos["end"])

                # Uses whichever opacity/word importance was strongest between the two words when merging the section of continous text
                current["opacity"] = max(current["opacity"], next_pos["opacity"])

                # Selects the words to be highlighted based on the new positions
                current["text"] = text[current["start"]:current["end"]]

            else:
                # If current word cannot merge with the next important word because they don't overlap, just add the current word dict to the list
                if next_pos["start"] > current["end"]: 
                    merged_positions.append(current)
                    # Moves to the next wor importance dict
                    current = next_pos
                else:
                    # If words overlap but have different colors (blue for real and red for fake), skip the word with the weaker importance or opacity
                    if next_pos["opacity"] > current["opacity"]:
                        current = next_pos
        
        # Adds the the current word segment
        merged_positions.append(current)  

    # Stores the highlighted text containing HTML tags
    result = []

    # Uses this to track pos in the text to start highlighting
    last_end = 0
    
    # Iterates over the dicts specifying highlighting positions and opacities (including merged consecutive sections of words)
    for pos in merged_positions:

        # If the next segment-to-highlight's start pos is greater than last_end, this means the previous text is NOT highlighted
        # so just add that previous plain text to the final result list (that will be joined into a string) and proceed
        if pos["start"] > last_end:
            result.append(text[last_end:pos["start"]])
        
        # Maps the segment dict's color label to the appropriate RGB color to use in in-line CSS HTML tags 
        color = "rgba(255, 0, 0," if pos["color"] == "red" else "rgba(0, 90, 156,"  # red (fake) or dodger blue (news)

        # Sets the value for the alpha opacity channel
        background_color = f"{color}{pos['opacity']})"
        
        # Appends the section of highlighted text with a span HTML tag and the inline CSS styling to the resulting highlighted text
        result.append(
            f"<span style='background-color:{background_color}; font-weight: bold'>"
            f"{pos['text']}</span>"
        )
        
        # Updates the last_end tracker to the final index of the highlighted section, to shift to the next part of the text
        last_end = pos["end"]
    
    # After it is done iterating through all highlighted segments, add any remaining non-highlighted text after last_end to the result list
    if last_end < len(text):
        result.append(text[last_end:])

    # Rejoins the result list containing HTML markup for highlighting into a string
    return "".join(result)



def detectWordBoundaries(text, word_start_pos, word):
    """
    A function ensuring that the highlighting func. is only matching whole words, and not parts of 
    words (e.g. avoid highlighting the word feature "hand" when iterating over the word "handle")

    It returns the start and end position if it's a valid word boundary (start and ende of word), but returns None otherwise

    Input Parameters:
        text (str): the whole text the word to highlight is part of
        word_start_pos (int): starting position of word (feature)
        word (str): the word feature that should be highlighted

    Output:
        the tuple of word_start_pos (int), word_end_pos (int): return positions only if word is indeed a word surrounded by a boundary
        None: returns None if there is no valid word boundary around this instance of the word substring
    """

    # Defines the series of punctuation characters for detecting word boundaries, such as space, exclamation mark, hyphen, etc.
    boundary_chars = set(' .,!?;:()[]{}"\n\t-')
    
    # Checks for word boundary at start of the word to highlight: passes check if position is 0 (start of text) 
    # # or if previous character is in the boundary_chars
    start_check = word_start_pos == 0 or text[word_start_pos - 1] in boundary_chars

    # Calculates the word's end position by adding the start idx to length of word
    word_end_pos = word_start_pos + len(word)

    # Checks for word boundary at end of word to highlight by comparing end pos to either length of whole text | if last pos is boundary char
    end_check = word_end_pos == len(text) or text[word_end_pos] in boundary_chars

    # If both start_check and end_check are set to True, returns the word start and end positions
    if start_check and end_check:
        return word_start_pos, word_end_pos

    # If the word is part of a larger word, returns None instead of the positions
    return None


def displayAnalysisResults(explanation_dict, container, news_text, feature_extractor, FEATURE_EXPLANATIONS):

    """
    Displays comprehensive LIME prediction analysis results including predivted label, prediction probabilities, 
    confidence scores, and feature importance charts.
    
    Input Parameters:

        explanation_dict (dict): dict containing LIME explanation results, including word and extra feature scores, 
                                 and HTML-marked up highlighted text
        
        container: the instance of the Streamlit app container to display the results in (passed in inside of app.py)
        
        news_text (str): the user's inputted text to generate prediction and LIME explanations for
        
        feature_extractor (an instance of the BasicFeatureExtractor class): for processing the inputted
                                                                            text to get semantic and linguistic features
        
        FEATURE_EXPLANATIONS (dict): natural language explanations of the different exra engineered non-word semantic
                                     and linguistic features
    """
    
    # Converts the news category label from 0 or 1 to text labels
    main_prediction = explanation_dict["main_prediction"]
    main_prediction_as_text = "Fake News" if explanation_dict["main_prediction"] == 1 else "Real News"
    
    # Extracts the [real, fake] probability array returned from LIME explainer function
    probs = explanation_dict["probabilities"]
    
    # Displays the prediction results based on LIME explainer output
    container.subheader("Text Analysis Results")
    # Write general predicted label
    container.write(f"**General Prediction:** {main_prediction_as_text}")
    # Write probabilities of being real and fake news
    container.write("**Confidence Scores:**")

    # Sets the color and boldness with markdown and HTML depending on the predicted label
    if main_prediction_as_text == "Real News":
        container.markdown(f"- <span style='color:dodgerblue; font-weight: bold'>Real News: {probs[0]:.2%}</span>", unsafe_allow_html=True)
        container.markdown(f"- <span style='color:red'>Fake News: {probs[1]:.2%}</span>", unsafe_allow_html=True)
    else:
        container.markdown(f"- <span style='color:dodgerblue'>Real News: {probs[0]:.2%}</span>", unsafe_allow_html=True)
        container.markdown(f"- <span style='color:red; font-weight: bold'>Fake News: {probs[1]:.2%}</span>", unsafe_allow_html=True)


    
    # Displays the highlighted text title using Markdown and inline CSS styling to adjust the size and padding
    st.markdown("""
        <div style='padding-top: 20px; padding-bottom:10px; font-size: 24px; font-weight: bold;'>
            Highlighted Text Section
        </div>
        """, unsafe_allow_html=True  # Reference: https://discuss.streamlit.io/t/unsafe-allow-html-in-code-block/54093
    )
    
    # Displays the explanation for what the color coding means in the highlighted text
    st.markdown("""
        <div style='padding-bottom:12px; padding-top: 12px; font-size: 18px; font-style: italic;'>
        Highlighted text shows the words (features) pushing the prediction towards <span style="color: dodgerblue; font-weight: bold;">real news</span> in blue 
        and to <span style="color: red; font-weight: bold;">fake news</span> in red.
        </div>
    """, unsafe_allow_html=True)
    
   
    # "Insert a multi-element container that can be expanded/collapsed."
    # Reference: https://docs.streamlit.io/develop/api-reference/layout/st.expander
    with st.expander("View Highlighted Text:"):
        # Formats the expandable scroll-box to show highlighted (blue=real, red=fake) text outputted by LIME Explainer using
        # inline CSS to allow y-scrolling and padding
        st.markdown("""
            <div style='height: 450px; overflow-y: scroll; border: 2px solid #d3d3d3; padding: 12px;'>
                {}
            </div>  
            """.format(explanation_dict["highlighted_text"]), unsafe_allow_html=True)
        
    # Adds more padding between the two charts
    st.markdown("""
    <style>
        .stColumn > div {
            padding: 0 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Adds title for bar charts for feature importance analysis
    container.subheader("Feature Importance Analysis")

    # Creates two columns for side-by-side bar charts for word features on the left and extra features on the right
    col1, col2 = container.columns(2)
    
    # First column: word features
    with col1:
        col1.write("### Top Word Features")

        # Extracts the top text-based features identified by LIME into a DataFrame for easier sorting and filtering
        word_features_df = pd.DataFrame(
            explanation_dict["word_features_list"],
            columns=["Feature", "Importance"]
        )

        # Applies a filter to select ONLY the words that pushed the classifier towards the main prediction (positive word scores)
        main_prediction_filtered_word_features_df = word_features_df[word_features_df["Importance"] > 0].copy()

        # Applies a filter to select ONLY words pushing towards the opposite class (negative word scores)
        opposite_prediction_filtered_word_features_df = word_features_df[word_features_df["Importance"] < 0].copy()

        # Sets the main prediction words graphs' titles based on whether main prediction is real or fake news
        if main_prediction == 0:  # Real news
            title = "Words pushing towards REAL NEWS"
        else: # Main prediction is fake news
            title = "Words pushing towards FAKE NEWS"
        
        # Sets the opposing prediction words graphs' titles based on whether main prediction is real or fake news
        if main_prediction == 0:  # Real news
            opposite_title = "Words pushing towards FAKE NEWS"
        else: # Main prediction is fake news
            opposite_title = "Words pushing towards REAL NEWS"

        # Extracts the top ten features of the words pushing towards the main prediction filtered DataFrame
        main_prediction_top_word_features_df =  main_prediction_filtered_word_features_df.nlargest(10, "Importance")

        # Extracts the top ten features of the words pushing towards the opposing prediction filtered DataFrame
        opposite_prediction_top_word_features_df = opposite_prediction_filtered_word_features_df.nlargest(10, "Importance")

        # Calculate the maximum importance value for pro-prediction word features and opposing-prediction word features
        # This is very important for scaling the y-axis on the two charts
        max_importance_score = 0

        # Calculates the max important score from the word features pushing towards the main prediction
        if len(main_prediction_top_word_features_df) > 0:
            max_importance_score = max(max_importance_score, main_prediction_top_word_features_df["Importance"].max()) # Updates max score if word importance max greater than 0

        # Calculates the ABSOLUTE max importance score from the opposing word feature scores (as they are negative)
        if len(opposite_prediction_filtered_word_features_df) > 0:
            opposite_max = opposite_prediction_filtered_word_features_df["Importance"].abs().max()
            # Update the max score if a greater value is found from the opposing word features
            max_importance_score = max(max_importance_score, opposite_max)

        # Adds a small offset to the top of the chart to make it look neater
        max_importance_score = max_importance_score * 1.1

        # Checks that there ARE important word features in the main prediction DataFrame
        if len(main_prediction_top_word_features_df) > 0:
            # Creates a bar chart for most important text features using the Altair visualization library
            # Q = specifies this feature/value is quantitative, N = specifies it is nominal/categorical
            # Reference: https://altair-viz.github.io/user_guide/generated/toplevel/altair.Chart.html#altair.Chart
            # Reference for sorting documentation: https://altair-viz.github.io/user_guide/generated/core/altair.EncodingSortField.html
            # How to create charts in Streamlit Reference: https://www.projectpro.io/recipes/display-charts-altair-library-streamlit
            # Use mark_bar to create bar_chart. Reference: https://altair-viz.github.io/user_guide/marks/index.html
            word_features_chart = alt.Chart(main_prediction_top_word_features_df).mark_bar().encode( 
                # Displays the categorical features (:N)/words on the x-axis
                x=alt.X(
                    "Feature:N", # N = categorical variable
                    sort=alt.EncodingSortField(
                        field="Importance",  # Sort by word importance (for main pred, importance goes from pos value to 0)
                        order="descending" 
                    ),
                    title="Word Feature",
                    axis=alt.Axis(
                        labelAngle=-45,  # Uses a rotation of 45 degrees for users to be able to read labels better
                        labelLimit=150,  # Maximum allowed pixel width of axis tick labels. Reference: https://altair-viz.github.io/user_guide/generated/core/altair.Axis.html
                        labelOverlap=False  # Prevent label overlap
                    )),
                # Plots the importance scores on y-axis 
                y=alt.Y( 
                        "Importance:Q", # Q means numerical value for Altair
                        title="Importance Strength",
                        # Reference: https://altair-viz.github.io/user_guide/generated/core/altair.Scale.html
                        scale=alt.Scale(domain=[0, max_importance_score]) # Set fixed scale for ease of comparison
                        ),
                # Sets blue bars for real news prediction, red for fake 
                color=alt.value("dodgerblue") if main_prediction == 0 else alt.value("red"), 
                # Adds "tooltips": explanations that appear when hovering above the bar
                tooltip=["Feature",
                        alt.Tooltip("Importance", title="Word Importance")]
            ).properties(
                title=title, # Set title and chart dimensions
                height=400,
                width=500
            ).configure_axis( # Reference: https://altair-viz.github.io/altair-viz-v4/user_guide/configuration.html
                labelFontSize=14,
                titleFontSize=16
            )

            # Displays the word features pushing towards the main prediction chart
            col1.altair_chart(word_features_chart , use_container_width=True)
        else:
            # If no important words have been found for pushing towards the main prediction, then print error message
            st.warning("No significant word features pushing the classifier towards the main prediction have been found.")


        # Creates the word features chart for words pushing towards the opposite (lower prob) prediction
        if len(opposite_prediction_top_word_features_df) > 0:
            
            # Modifies the original DF to get absolute importance scores for easier plotting in desc order from 0 to high
            opposite_df_for_chart = opposite_prediction_top_word_features_df.copy()

            # Applies abs func to whoel Importance col
            opposite_df_for_chart["Importance"] = opposite_df_for_chart["Importance"].abs()

            # Sorts the (now absolute) importance values in desc order
            opposite_df_for_chart = opposite_df_for_chart.sort_values("Importance", ascending=False)

            # Creates the chart with the sorted abs values pushing towards the opposite prediction
            opposite_word_features_chart = alt.Chart(opposite_df_for_chart).mark_bar().encode(
                x=alt.X(
                    "Feature:N", # x-axis = categorical value (word feature)
                    sort=alt.EncodingSortField(
                        field="Importance",
                        order="descending"
                    ),
                    title="Word Feature",
                    axis=alt.Axis(
                        labelAngle=-45, # Rotates labels for readability
                        labelLimit=150, # Set label length to 150 pixels
                        labelOverlap=False # Don't let labels overlap
                    )),
                y=alt.Y( # y-axis = word importance score (quantitative, number)
                    "Importance:Q",
                    title="Importance Strength",
                    scale=alt.Scale(domain=[0, max_importance_score])
                    ),
                # Uses the opposite color to the main prediction
                color=alt.value("red") if main_prediction == 0 else alt.value("dodgerblue"),
                tooltip=["Feature",
                        alt.Tooltip("Importance", title="Word Importance")]
            ).properties(
                title=opposite_title, # Adds title and dimensions
                height=400,
                width=500
            ).configure_axis(
                labelFontSize=14, # Sets label and title font szzies
                titleFontSize=16
            )
        
            # Displays the opposite word features chart
            col1.altair_chart(opposite_word_features_chart, use_container_width=True)

        else:

            # Prints an error message if no significant words pushing towards the opposing prediction are found
            st.warning("No significant word features pushing the classifier away from the main prediction have been found.")
        
    # The second column is for displaying a bar chart with the importance scores for non-word features
    with col2:

        col2.write("### Top Extra Semantic and Linguistic Features")
        
        # Creates a DataFrame from the extra features list returned by the LIME explainer function
        extra_features_df = pd.DataFrame(
            explanation_dict["extra_features_list"],
            columns=["Feature", "Importance"]
        )
        
        # Sorts the features by their absolute importance for easier plotting
        extra_features_df["Absolute Importance"] = extra_features_df["Importance"].abs()
        # Extracts 10 largest features (all of the features) by absolute importance
        extra_features_df = extra_features_df.nlargest(10, "Absolute Importance")
        
        # Adds the original feature name, not the underscored programmatic title, before mapping to the explanation labels on the chart
        extra_features_df["Original Feature"] = extra_features_df["Feature"]
        
        # Map feature column variables to user-readable strings 
        feature_name_mapping = {
            "exclamation_point_frequency": "Exclamation Point Usage",
            "third_person_pronoun_frequency": "3rd Per. Pronoun Usage",
            "noun_to_verb_ratio": "Noun/Verb Ratio",
            "cardinal_named_entity_frequency": "Number Usage",
            "person_named_entity_frequency": "Person Name Usage",
            "nrc_positive_emotion_score": "Positive Emotion",
            "nrc_trust_emotion_score": "Trust Score",
            "flesch_kincaid_readability_score": "Readability Grade",
            "difficult_words_readability_score": "Difficult Words",
            "capital_letter_frequency": "Capital Letter Usage"
        }
        
        # Maps the features to their more readable names listed in the dictionary above
        extra_features_df["Feature"] = extra_features_df["Feature"].map(feature_name_mapping)
        
        # Maps the explanations to their globally-stored natural language explanation for users
        extra_features_df["Explanation"] = extra_features_df["Original Feature"].map(FEATURE_EXPLANATIONS)
        
        # Creates a bar chart with more tooltips for explaining what features mean to users when they hover over a bar
        extra_features_chart = alt.Chart(extra_features_df).mark_bar().encode(
            x=alt.X("Importance:Q", title="Importance Strength"),
            y=alt.Y("Feature:N",
                    sort=alt.EncodingSortField(
                        field="Absolute Importance", # Sorts features by ABSOLUTE value of importance in desc order
                        order="descending"
                    ),
                    axis=alt.Axis(
                        labelFontSize=14 # Sets the feature labels font size
                    )), 
                color=alt.condition(
                # Checks if the importance score pushes in the same direction as the predicted class
                alt.datum.Importance > 0,
                alt.value("dodgerblue") if main_prediction == 0 else alt.value("red"), # If same dir as pred class
                alt.value("red") if main_prediction == 0 else alt.value("dodgerblue") # If NOT same dir as pred class
            ),
            tooltip=["Feature", "Importance", "Explanation"]  # Add tooltip showing feature and importance if user hovers over the bar 
        ).properties(
            title="Top 10 Extra Features", # Sets title and dimensions
            height=400,
            width=500
        )
        
        # Displays the extra features chart in the second column
        col2.altair_chart(extra_features_chart, use_container_width=True) # Make span across whole container
        
        # Adds a legend explanation for the color-coded bar charts and highlighted text
        container.markdown(f"""
            **Legend:**
            - 🔵 **Blue bars**: Features pushing towards real news classification
            - 🔴 **Red bars**: Features pushing towards fake news classification
            
            The length of each bar and color represent how strongly this feature
            in the news text pushes the classifier towards a REAL or FAKE prediction.
            For more details about these features' distributions and patterns
            in the real vs fake training data this model was trained on,
            please click below to expand the explanations, or go to the 'Key Pattern Visualizations''
            tab to view charts showing the global frequency counts for these features in 
            the global training daa.
        """)
        
        # Adds an expander outlining the feature importance scores in more detail
        with col2.expander("*View More Detailed Feature Score Information*"):
            
            # Iterates over the extra features to explain each one
            for index, row in extra_features_df.iterrows():

                # Determines the importance color and explanation based on the prediction and score value
                if row["Importance"] > 0: # Same direction of word importance as main prediction
                    if main_prediction == 0: # Main pred AND score is real news
                        importance_color = "dodgerblue"
                        importance_explanation = "(pushing towards real news)"
                    else:
                        importance_color = "red" # Main pred AND score are fake news
                        importance_explanation =  "(pushing towards fake news)"    
                elif row["Importance"] < 0: # Different dir of word importance as main pred
                    if main_prediction == 0: # Main pred is real BUT word pushes towards fake news
                        importance_color = "red"
                        importance_explanation = "(pushing towards fake news)"
                    else: # Main pred is fake BUT word pushes towards real news
                        importance_color = "dodgerblue"
                        importance_explanation =  "(pushing towards real news)"    
                else:
                    importance_color = "grey"
                    importance_explanation =  "Neutral - there was no significant impact on prediction."  
                
                # Adds explanation from FEATURE_EXPLANATIONS about the feature and main patterns associated with it in the training data
                container.markdown(f"""
                    **{row["Feature"]}**
                    - Impact on Classification: <span style='color:{importance_color}'>{row["Importance"]:.4f} {importance_explanation}</span>
                    - {FEATURE_EXPLANATIONS[row["Original Feature"]]}
                    ---
                """, unsafe_allow_html=True)
                
