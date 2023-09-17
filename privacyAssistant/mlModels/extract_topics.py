# %% [markdown]
# # In this script, you can find how to extract features (topics)

# %% [markdown]
# # Import required packages

# %%
import pandas as pd
import numpy as np
import math
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from onlineAssistant.settings import BASE_DIR
import os

# %%

# %%
# "dataset_dir" is the folder which has the Train and Test datasets
# you should create this folder uploading datasets in the repo.
dataset_dir = str(BASE_DIR) + '/privacyAssistant/dataset_dir'
pickle_dir = str(BASE_DIR) + '/privacyAssistant/pickle_dir'



df_train = pd.read_csv(dataset_dir +"/df_train.csv")
df_test = pd.read_csv(dataset_dir +"/df_test.csv")


# %% [markdown]
# # Topic Modeling using Non-negative Matrix Factorization (NMF) <br>
# \begin{equation}
# X \approx W \times H
# \end{equation}
# <BR>
# X: (image, tag), W: (image, topic), H: (topic, tag) matrices <br>
# [sklearn - NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html) <br>
# # Term Frequency-Inverse Document Frequency (TF-IDF) <br>
# **Term Frequency:** the number of times a tag appears in an image. <br>
# **Inverse Document Frequency:** total images in Training data over number of images with tag. <br>
# [sklearn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) <br>

# %%
# Use TF-IDF to weight tags
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(df_train.cleaned_tags).toarray()
print("TFIDF's features: ", tfidf_vectorizer.get_feature_names_out())

# %%
number_of_topics = 20
nmf_model = NMF(n_components=number_of_topics, init='random', random_state=5)
nmf_features = nmf_model.fit_transform(X)

# %%
# Show tags in the Training data with topic-association
components_df = pd.DataFrame(nmf_model.components_, columns = tfidf_vectorizer.get_feature_names_out())
components_df


# %%
# Show the top 20 most associated tags with topics
individual_topic_dict = {}
for topic in range(components_df.shape[0]):
    tmp = components_df.iloc[topic]
    individual_topic_dict[topic] = list(tmp.nlargest(20).keys())

df_train_topic_tag = pd.DataFrame.from_dict(individual_topic_dict)
df_train_topic_tag


# %% [markdown]
# # Predict Topic Distributions of Test Data

# %%
# Transform the TF-IDF
X_test = tfidf_vectorizer.transform(df_test.cleaned_tags)
nmf_features_test = nmf_model.transform(X_test)
pd.DataFrame(nmf_features_test)

# %%
# Prepare input files for the training and the test
train_input = pd.DataFrame(nmf_features).to_numpy()
test_input = pd.DataFrame(nmf_features_test).to_numpy()
train_label = np.array(df_train.label)
train_label = np.array(list(map(lambda x: math.floor(x), train_label)))
test_label = np.array(df_test.label)

train_input.shape, test_input.shape, train_label.shape, test_label.shape

model_save_path = os.path.join(BASE_DIR, 'privacyAssistant/mlModels/nmf_model.pkl')
joblib.dump(nmf_model, model_save_path)
model_save_path = os.path.join(BASE_DIR, 'privacyAssistant/mlModels/tfidf_vectorizer.pkl')
joblib.dump(tfidf_vectorizer, model_save_path)