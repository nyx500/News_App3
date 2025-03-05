# app.py
# Main app starting point

# Imports the required libraries
# Streamlit for app creating
import streamlit as st

# Set Streamlit page configuration e.g. page title
# Reference: https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config
st.set_page_config(
    page_title="Fake News Detection App",
    layout="centered"   
)

# A library for extracting news articles from URLs
from newspaper import Article
# For loading in the graphs and charts showing global fake vs real news patterns
import matplotlib.pyplot as plt
# For CalibratedClassifierCV and Passive-Aggressive model loading
import joblib
# For downloading the fastText model
import gdown
import fasttext
# Imports the custom functions from lime_functions.py for generating LIME explanations
from lime_functions import BasicFeatureExtractor, explainPredictionWithLIME, displayAnalysisResults

# Maps DataFrame extra feature names to explanations about their tendencies + patterns in train data
FEATURE_EXPLANATIONS = {
    "exclamation_point_frequency": 
        """Normalized exclamation marks frequency counts. Higher raw frequencies of exclamation marks in text 
        can indicate more emotional or sensational writing, and thus were more strongly associated 
        with fake news in the training data.""",
    
    "third_person_pronoun_frequency": 
        """Normalized frequency of third-person pronouns (he, she, they, etc.). Higher raw scores might indicate a more story-like 
        narrative style. Therefore, the more positive the original score scores, the more likely the text was to be fake news.""",
    
    "noun_to_verb_ratio": 
        """Ratio of nouns to verbs. Higher raw values (more nouns) suggest more descriptive rather than action-focused writing. 
        Higher scores (more nouns to verbs) were thus associated more with real news than fake news in training data. 
        Negative values, with more verbs, were associated with fake news more than real news, perhaps due to contributing 
        towards a more fast-paced, action-based style.""",
    
    "cardinal_named_entity_frequency": 
        """Normalized frequency of numbers and quantities. Higher raw scores indicate a higher level of specific details, and they were
        more associated with real news in the training data. Fake news contained less numerical facts on the whole.""",
    
    "person_named_entity_frequency": 
        """Normalized frequency of PERSON named entities. Shows how person-focused the text is, higher scores more 
        associated with fake news, showing that disinformation campaigns (at least based on the training data),
        was closely tied to attempts to harm a person's reputation through different propaganda campaigns.""",
    
    "nrc_positive_emotion_score": 
        """Measure of the positive emotional content using the NRC lexicon. Higher raw values indicate more words associated with positive emotions, 
        and a more positive tone was more closely associated more with real news than fake news in the training data.""",
    
    "nrc_trust_emotion_score": 
        """The normalized count of trust-related words using NRC lexicon. Higher raw values suggest more credibility-focused language, 
        and were more associated with real news than fake news. Fake news contained considerably lower scores for trust-related words
        in the training data.""",
    
    "flesch_kincaid_readability_score": 
        """U.S. grade level required to understand the text. A higher raw score indicates more complex writing, which was 
        associated more with real news in the training data, while fake news in general used simpler language.""",
    
    "difficult_words_readability_score": 
        """Count of complex words not included in the Dall-Chall word list. Higher raw values indicate more sophisticated vocabulary, 
        and this was associated more with real news in the training data.""",
    
    "capital_letter_frequency": 
        """Normalized frequency of capital letters. Higher raw values might indicate more emphasis and acronyms. 
        Greater capital letter usage was associated more with real news in training data."""
}

 # Defines a reusable warning message about how to use the app properly to users for the 'How it Works' tab
warning_message = """
**Some Guidance on App Usage:** \n 
- The LIME explanation algorithm used to determine word importance scores functions by removing random words from the text, and then calculating the impact this had on the final prediction.\n  
- However, the bar charts showing word features pushing towards either the main prediction or against it can carry little meaning **out of context**.\n  
- Therefore, it is recommended to also inspect how the word appears in the entire **highlighted text**. E.g. the word 'kind' can mean sympathetic and helpful, but it can also be part of the colloquial phrase 'kind of' and thus carry a different meaning!  
- **Please remember that the final probabilities are a result of interactions between different features, and not the result of one feature alone.** \n 
"""

# Another warning msg for the first 2 (URL and text input) tabls
warning_message_for_first_two_tabs = """
        **Please remember that the final probabilities are a result of interactions between different features, and not the result of one feature alone.**
        \n Go to the *How it Works* tab for a more detailed explanation of how to interpret the word important scores.
"""

# Loads the trained fastText model
@st.cache_resource # Save it for quicker loading next time
def load_fasttext_model():
    # Load the pre-trained fastText from Google Drive
    url = "https://drive.google.com/uc?id=1uO8GwNHb4IhqR2RNZqp1K-FdmX6FPnDQ"
    local_path =  "/tmp/fasttext_model.bin"
    gdown.download(url, local_path, quiet=False)
    return fasttext.load_model(local_path)

# Load CalibratedClassifierCV model (wrapping Passive-Aggressive Classifier to output probabilities), trained on 4 datasets combined
@st.cache_resource 
def load_pipeline():
    return joblib.load("all_four_calib_model.pkl")

# Load the pre-trained scikit-learn StandardScaler for scaling extra feature ranges, trained on the four-dataset combined dataset
@st.cache_resource
def load_scaler():
    return joblib.load("all_four_standard_scaler.pkl")

# Instantiates a text feature extractor for engineered semantic and linguistic features
feature_extractor = BasicFeatureExtractor()

# Load the models and show progress
with st.spinner("Loading fake news detection model..."):
    pipeline = load_pipeline()

with st.spinner("Loading pre-fitted feature scaler..."):
    scaler = load_scaler()

with st.spinner("Loading fastText embeddings model..."):
    fasttext_model = load_fasttext_model()

# Sets the whole app's title
st.title("Fake News Detection App")

# Creates tabs for news text classification, visualizations, app about information
tabs = st.tabs(["Enter News as URL", "Paste in Text Directly", "Key Pattern Visualizations",
                "Word Clouds: Real vs Fake", "How it Works..."])

# A bit of custom CSS that makes the main app container slightly wider, to stop all content being all cramped together
# Reference: https://discuss.streamlit.io/t/modify-size-margin-of-the-main-page-container/21269/2
with st.container():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1000px;
            padding: 3rem;
            margin: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # First tab: News Input as URL
    with tabs[0]:

        # Sets the tab title
        st.header("Paste URL to News Text Here")

        # Sets the text area for user to enter a news URL
        url = st.text_area("Enter news URL for classification", placeholder="Paste your URL here...", height=68)
        
        # Adds a Streamlit slider to let users select the number of perturbed samples for LIME to generate feature explanations
        num_perturbed_samples = st.slider(
            "Select the number of perturbed samples for explanation",
            min_value=25,
            max_value=500,
            value=50,  # Default value
            step=25, # Step size of 25
            help="Increasing the number of samples will make the outputted explanations more accurate but may take longer to process!"
        )
        
        # Explains slider and how feature importance works to users
        st.write("The more perturbed samples you choose, the more accurate the explanation will be, but it will take longer to output.")
        # Displays a brief warning message about how to interpret word features scores
        st.warning(warning_message_for_first_two_tabs, icon='🚩')
        
        # Creates interactive button to classify text and descriptor for this specific tab
        if st.button("Classify", key="classify_button_url"): # If URL classificationbutton pressed...
            
            # Checks if news URL input is not empty
            if url.strip():  
                # Tries to extract the news text from the URL using the newspaper3k library (can fail, so adds a try-except block here)
                try:
                    # Displays progress
                    with st.spinner("Extracting news text from URL..."):

                        # Uses the newspaper3k to scrape the news article
                        # Reference: https://newspaper.readthedocs.io/en/latest/
                        article = Article(url)
                        article.download()
                        article.parse()
                        # Extract the article's text content
                        news_text = article.text
                        
                        # Displays the scraped original text in an expander
                        with st.expander("View the Original News Text"):
                            st.text_area("Original News Text", news_text, height=300)
                        
                        # Generates the prediction and LIME explanation using the custom func in lime_functions.py
                        with st.spinner("Analyzing text..."):
                            explanation_dict = explainPredictionWithLIME(
                                fasttext_model, # fastText model
                                pipeline, # PAC model wrapped in CalibratedClassifierCV
                                scaler, # Pre-fitted StandardScaler
                                news_text, # The news text
                                feature_extractor, # Feature extractor instance for engineered features
                                num_features=50, # Num of word features LIME generates with importance scores
                                num_perturbed_samples=num_perturbed_samples # User-inputted num of perturbed samples for LIME to generate
                            )
                            
                            # Displays the highlighted text, charts and LIME explanations
                            displayAnalysisResults(explanation_dict, st, news_text, feature_extractor, FEATURE_EXPLANATIONS)

                # If could not extract either news article or LIME scores
                except Exception as e:
                    st.error(f"Sorry, but there was an error downloading and processing the news text: {e}. Please try a different text!")
            # If empty URL text input field
            else:
                st.warning("Warning: Please enter some valid news text for classification!")

    # Second tab: News Input copied and pasted in directly as text
    with tabs[1]:

        st.header("Paste News Text In Here Directly")
        news_text = st.text_area("Paste the news text for classification", placeholder="Paste your news text here...", height=300)
        
        # Add the same slider to let users select the number of perturbed samples for LIME explanations
        num_perturbed_samples = st.slider(
            "Select the number of perturbed samples to use for the explanation",
            min_value=25,
            max_value=500,
            value=50,  # Default value is 50, it is a "sweet spot" between time and accuracy of explanations
            step=25,
            help="Warning: Increasing the number of samples will make the outputted explanations more accurate but may take longer to process!"
        )
        
        # Explains the slider settings and how feature importance works to users
        st.write("The more perturbed samples you choose, the more accurate the explanation will be, but it will take longer to compute.")
        # Displays the same warning message as in the first tab
        st.warning(warning_message_for_first_two_tabs, icon='🚩')
        
        # If user presses the classifier button on this tab, then proceed
        if st.button("Classify", key="classify_button_text"):
            # Checks if copy-and-paste text input is not empty
            if news_text.strip():  
                # Try to process news text with LIME
                try:
                    # Uses the entered text as news text directly for classification
                    with st.spinner(f"Analyzing text with {num_perturbed_samples} perturbed samples..."):
                        explanation_dict = explainPredictionWithLIME(
                            fasttext_model, # fastText model
                            pipeline, # PAC model wrapped in CalibratedClassifierCV
                            scaler, # Pre-fitted StandardScaler
                            news_text, # The news text
                            feature_extractor, # Feature extractor instance for engineered features
                            num_features=50, # Num of word features LIME generates with importance scores
                            num_perturbed_samples=num_perturbed_samples # User-inputted num of perturbed samples for LIME to generate
                        )
                        
                        # Displays the highlighted text and LIME feature importance charts
                        displayAnalysisResults(explanation_dict, st, news_text, feature_extractor, FEATURE_EXPLANATIONS)
                    
                # If could not process the text, informs the user with an error message         
                except Exception as e:
                    st.error(f"Sorry, but there was an error while analyzing the text: {e}")

            else:
                # If news text field was empty
                st.warning("Warning: Please enter some valid news text for classification!")

    # Third tab: Visualizations and charts of REAL vs FAKE news patterns in the training datast
    with tabs[2]:

        st.header("Key Patterns in the Training Dataset: Real (Blue) vs Fake (Red) News")

        st.write("These visualizations show the main trends and patterns between real and fake news articles in the training data.")

        # Adds space between graphs
        st.write("") 
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        

        # Capital Letter Usage
        st.subheader("Capital Letter Usage")
        caps_img = plt.imread("all_four_datasets_capitals_bar_chart_real_vs_fake.png")
        st.image(caps_img, caption="Mean number of capital letters in real vs fake news", use_container_width=True)
        st.write("Real news tended to use more capital letters, perhaps due to including more proper nouns and technical acronyms.")

        st.write("") 
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # Exclamation Point Usage
        st.subheader("Exclamation Point Usage")
        exclaim_img = plt.imread("all_four_datasets_exclamation_points_bar_chart_real_vs_fake.png")
        st.image(exclaim_img, caption="Frequency of exclamation points in real vs fake news", use_container_width=True)
        st.write("Fake news tends to use more exclamation points, possibly suggesting a more sensational and inflammatory writing.")

        st.write("") 
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # Third-Person Pronoun Usage
        st.subheader("Third Person Pronoun Usage")
        pronouns_img = plt.imread("all_four_datasets_third_person_pronouns_bar_chart_real_vs_fake.png")
        st.image(pronouns_img, caption="Frequency of third-person pronouns in real vs fake news", use_container_width=True)
        st.write("Fake news often uses more third-person pronouns (e.g him, his, her), which could indciate a more 'storytelling' kind of narrative style.")

        st.write("") 
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # Noun-to-Verb Ratio
        st.subheader("Noun-to-Verb Ratio")
        emotions_img = plt.imread("all_four_datasets_noun_to_verb_ratio_bar_chart_real_vs_fake.png")
        st.image(emotions_img, caption="Noun-to-Verb Ratio: Real vs Fake News", use_container_width=True)
        st.write("In the training data, real news tended to have slightly more nouns than verbs than fake news.")

        st.write("") 
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
        # Emotion counts
        st.subheader("Emotional Content using NRC Emotion Lexicon")
        emotions_img = plt.imread("all_four_datasets_emotions_bar_chart_real_vs_fake.png")
        st.image(emotions_img, caption="Emotional content comparison between real and fake news", use_container_width=True)
        st.write("Fake news (in this dataset) often showed lower positive emotion scores and fewer trust-based emotion words than real news.")

        st.write("") 
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # Named Entity PERSON counts
        st.subheader("Named Entity PERSON Frequency Counts")
        emotions_img = plt.imread("all_four_datasets_person_named_entities_bar_chart_real_vs_fake.png")
        st.image(emotions_img, caption="PERSON named entity count for fake vs real news", use_container_width=True)
        st.write("Fake news (in this dataset) often contained more references to PERSON named entities than real news.")

        st.write("") 
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # Named Entity CARDINAL counts
        st.subheader("Named Entity CARDINAL (i.e. numbers) Frequency Counts")
        emotions_img = plt.imread("all_four_datasets_cardinal_named_entities_bar_chart_real_vs_fake.png")
        st.image(emotions_img, caption="CARDINAL (numbers) named entity count for fake vs real news", use_container_width=True)
        st.write("Fake news tended to contain less numerical data (i.e. lower CARDINAL named entity frequencies) than real news.")

        st.write("") 
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # Flesch-Kincaid Readability Grade
        st.subheader("Flesch-Kincaid U.S. Readability Grade Level")
        emotions_img = plt.imread("all_four_datasets_flesch_kincaid_readability_bar_chart_real_vs_fake.png")
        st.image(emotions_img, caption="Flesch-Kincaid avg. U.S. grade level (readability) for fake vs real news", use_container_width=True)
        st.write("Real news tended to have a slightly higher U.S. grade level, indicating more complex language, than fake news.")

        st.write("") 
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # Difficult-Words score
        st.subheader("Difficult Words Score")
        emotions_img = plt.imread("all_four_datasets_difficult_words_score_bar_chart_real_vs_fake.png")
        st.image(emotions_img, caption="Normalized 'Difficult Words' scores for fake vs real news", use_container_width=True)
        st.write("Real news tended to contain more complex words than fake news.")

        st.write("") 
        st.write("")
        st.write("")
        st.write("")
        st.write("")


        # Adds an expander with more detailed explanation
        with st.expander("📊 Details about these visualizations"):
            st.markdown("""
            These charts are created on the basis of a detailed data analysis of four benchmark fake news datasets used for training the model:
            WELFake, Constraint (COVID-19 data), PolitiFact (political news), and GossipCop (entertainment and celebrity news). 
            The charts display the NORMALIZED frequencies, e.g. for exclamation marks and capital use: the 
            raw frequencies have been divided by the text length (in words) to account for the differences in the lengths of the different news texts.
            
            ### Some of the Main Differences between Real News and Fake News Based on the Data Analysis:
            
            - **Capital Letter Frequencies:** Higher frequencies were found in real news, perhaps due to the greater usage of proper nouns and techical acronyms
            - **Third-person Pronoun Frequencies:** Third-person pronouns were more frequenty encountered in fake news in these datasets, suggesting storytelling-like narrative style and person-focused content
            - **Exclamation Point Frequencies:** These were more frequent in fake news too, pointing towards a sensational inflammatory style
            - **Emotion (Trust and Positive) Features:** The words used in fake news tended to have much less positive emotional connotations and reduced trust scores.
            - **Named-Entity (PERSON and CARDINAL) Frequencies:** While fake news contained more PERSON references, real news tended to contain more CARDINAL (number) references
                    to quantitative entities.
            - **Readability Scores:** On the whole, real news tended to contain more complex words and language than fake news
            
            Disclaimer: These patterns were specific to THESE four datasets, but they should be considered in combination with other features
            (i.e. the word feature importance), as well as remembering that more recent fake news may exhibit different trends, particularly
            given the rapid analysis of propaganda and disinformation strategies
            
            """)


    # Fourth tab: Word Clouds showing Named Entities Exclusive to Real vs Fake News
    with tabs[3]:
        st.header("Most Common Named Entities: Exclusive Entities found in Real vs Fake News")
        st.write("These word clouds visualize the most frequent named entities (e.g. people, organizations, countries) in real vs fake news articles in the training data. The size of each word is proportional to how frequently it appears.")
        
        st.subheader("Named Entities Appearing ONLY in Real News and NOT in Fake News")
        # Read in the image using matplotlib.pyplot.imread functionality
        real_cloud_img = plt.imread("combined_four_set_training_data_real_news_named_entities_wordcloud.png")
        st.image(real_cloud_img, caption="Most frequent entities exclusive to real news", use_container_width=True)
        
        # Adding spaces
        st.write("") 
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        
        st.subheader("Named Entities Appearing ONLY in Fake News and NOT in Real News")
        fake_cloud_img = plt.imread("combined_four_set_training_data_fake_news_named_entities_wordcloud.png")
        st.image(fake_cloud_img, caption="Most frequent entities exclusive to fake news", use_container_width=True)
    
        st.write("") 
        st.write("")
        
        # Adding an explanation for how the word clouds use sizes colors
        with st.expander("☁️ Word Cloud Explanation"):
            st.write("""
            The size of the word reflects how frequently it occurred in each dataset.
            The colors are only used for readability - they don't carry any additional meaning.
            """)

    with tabs[4]:
        st.header("❓ How Does the App Work?")
        
        st.write("""
        The LIME algorithm (Local Interpretable Model-agnostic Explanations) is used here to explain the
        specific prediction made for a piece of news (i.e. whether the news text is classed as real or fake news).
                
        Let's get a glimpse into the general intuition behind how this technique works.
                
        """)
        
        st.subheader("⚙️ The Main Idea Behind LIME")
        st.write("""
        Whenever this app analyzes a news text, it doesn't just tell you if the news is "fake news" or "real news". The main concept behind
        LIME is to explain which features (e.g. words, certain punctuation patterns) of the text led the model to make the outputted decision.
        As such, highlights WHICH word features, or more high-level semantic and linguistic features (such as use of certain punctuation marks)
        , in the news text led to the outputted classification. Furthermore, the algorithm also outputs the probability of news being fake,
        rather than a simple label, so that you can get an insight into the certainty of the classifier.
        """)
        
        st.subheader("🍋‍🟩 How Does LIME Generate the Explanations?")
        st.write("""
        LIME removes certain words or linguistic features in the news text one-by-one, and runs the trained machine-learning model to see
        how the outputted probabilities change when the text has been slightly changed.

        (a) LIME randomly removes words or linguistic features from the news input
        (b) It then runs the altered versions of the news texts through the classifier and records how much changing these individual features
        has impacted the final prediction
        (c) If changing a specific feature (e.g. emotion score) has a big impact on the final predicted probability, this feature is then assigned a higher importance
        score. The importance scores are then visualized using bar graphs and highlighted texts. Red color-coding means that this feature is associated more
        with fake news, and blue color-coding means this feature makes the text more likely to be real news.
        """)
        
        st.subheader("📈 Which extra features (apart from words) have been included for making predictions?")
        st.write("""
        This model classifies news articles based on the specific features that were found to be the most useful for discriminating 
        between real and fake news based on an extensive exploratory data analysis:

        - Individual words that push a prediction to either real news or fake news
        - Use of punctuation e.g. exclamation marks, capital letters
        - Grammatical patterns such as noun-to-verb ratio
        - Frequencies of PERSON and CARDINAL (number) named entities
        - Trust and positive emotion scores (using the NRC Emo Lexicon)
        - Text readability scores (how hard the text is to read), e.g. how many difficult words are used, U.S. Grade readability level
        """)
        
        with st.expander("⁉️ Why Were THESE Particular Features Chosen?"):
            st.write("""
            These features were engineered based on a detailed exploratory analysis focusing on the key differences between real and fake news
            over four benchmark datasets: WELFake (general news), Constraint (COVID-19 related health news), PolitiFact (political news),
            and GossipCop (celebrity and entertainment news).
            
            - Fake news is often associated with a more sensational style (e.g. using more exclamation points) than real news, and more "clickbaity" language
            - Real news tends to use more nouns than verbs, as well as more references to numbers, signalling a more factal style
            - Narrative style (e.g. using more third-person pronouns indicates a more "storytelling" style) can also be a key indicator of fake news
            - Text readability and complexity can also help the classifier distinguish between real and fake news,
            as fake news tends to be easier to digest and less challenging.
            """)

        # Places the warning message inside a Streamlit warning container
        st.warning(warning_message, icon='🚩')
            
        st.subheader("😐 Disclaimer: Limitations of the Model")
        st.markdown("""
            Please bear in mind that the strategies for producing fake news/propaganda are always evolving rapidly, especially due to the rise of generative AI.
            The patterns highlighted here are based on THIS specific training data from four well-known fake news datasets; however,
            they may not apply to newer forms of disinformation!  As a result, it is also strongly recommended to
            use fact-checking and claim-busting websites to check out whether the sources of information are legitimate.
            <br>
            <br>
            The model used to classify fake news here obtained 93% accuracy and F1-score on the training data composed of four different
            dataset from different domains, therefore its predictions are not perfect.
        """, unsafe_allow_html=True)
        
