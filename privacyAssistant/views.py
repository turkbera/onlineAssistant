import os
import joblib
from django.shortcuts import render
from onlineAssistant.settings import BASE_DIR
from .modelPURE.get_tags import get_tags_from_photo
import numpy as np

# Load the saved NMF model
model_save_path = os.path.join(BASE_DIR, 'privacyAssistant/modelPURE/nmf_model.pkl')
nmf_model = joblib.load(model_save_path)

# Load the TF-IDF vectorizer
tfidf_vectorizer_save_path = os.path.join(BASE_DIR, 'privacyAssistant/modelPURE/tfidf_vectorizer.pkl')
tfidf_vectorizer = joblib.load(tfidf_vectorizer_save_path)

# Load the Random Forest classifier
classifier_rf_save_path = os.path.join(BASE_DIR, 'privacyAssistant/modelPURE/classifier_rf.pkl')
classifier_rf = joblib.load(classifier_rf_save_path)

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
        predict = classifier_rf.predict(photo_topics)
        # Pass the extracted tags and topics to the template context
        prediction_privacy = "private" if predict[0] == 0 else "public" 
        context = {
            'tags': predicted_tags,
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