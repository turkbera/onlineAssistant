import os
import joblib
from sklearn import metrics
import pandas as pd
from django.shortcuts import render
from onlineAssistant.settings import BASE_DIR
from .mlModels.get_tags import get_tags_from_photo
from .mlModels.generate_explanations import generate_exp
import numpy as np
import math
import pickle

# Load the saved NMF model
model_save_path = os.path.join(BASE_DIR, 'privacyAssistant/mlModels/nmf_model.pkl')
nmf_model = joblib.load(model_save_path)

# Load the TF-IDF vectorizer
tfidf_vectorizer_save_path = os.path.join(BASE_DIR, 'privacyAssistant/mlModels/tfidf_vectorizer.pkl')
tfidf_vectorizer = joblib.load(tfidf_vectorizer_save_path)

# Load the Random Forest classifier
classifier_rf_save_path = os.path.join(BASE_DIR, 'privacyAssistant/mlModels/classifier_rf.pkl')
classifier_rf_fit = joblib.load(classifier_rf_save_path)

# Function to extract topics from tags
def extract_topics_from_tags(tags):
    # Transform the tags using the TF-IDF vectorizer
    tags = " ".join([tag for tag in tags])
    tags_tfidf = tfidf_vectorizer.transform([tags])
    # Use the NMF model to extract topics
    topics = nmf_model.transform(tags_tfidf)
    return topics

def index(request):
    if request.method == 'POST':
        # Retrieve the uploaded photo from the request
        photo = request.FILES['photo']

        # Process the photo using the get_tags_from_photo function
        photo_bytes = photo.read()
        predicted_tags = get_tags_from_photo(photo_bytes)

        # Convert tags to a list of strings
        # Extract topics from tags using the NMF model
        

        photo_topics = extract_topics_from_tags(predicted_tags)
        predict = classifier_rf_fit.predict(photo_topics)
        topics = 20
        my_list = [str(i) for i in np.arange(topics)]
        input_columns = list(map(lambda orig_string: 'topic ' + orig_string, my_list))
        df_topics = pd.DataFrame(photo_topics, columns = input_columns)
        
         # Pass the extracted tags and topics to the template context
        topics, category = generate_exp(df_topics, predict[0])
        prediction_privacy = "private" if predict[0] == 0 else "public" 
        context = {
            'tags': predicted_tags,
            'category': category,
            'predict': prediction_privacy
        }

        return render(request, 'privacyAssistant/index.html', context)

    return render(request, 'privacyAssistant/index.html')

def about(request):
    email_address = 'turk.mbera@gmail.com'  # Replace this with your email address
    context = {
        'email_address': email_address
    }
    return render(request, 'privacyAssistant/about.html', context)