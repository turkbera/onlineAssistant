# %%


# %%
import pandas as pd
import numpy as np
import pickle
from IPython.display import display
import shap
from .classify_images import show_topics_contributions
import ast
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# %%


# %% [markdown]
# # Load required files

# %%
# you can change the path of the directory based on where you saved the repo.
dataset_dir = '/home/bera/Desktop/uzco/peak/dataset_dir'
pickle_dir = '/home/bera/Desktop/uzco/peak/pickle_dir'


# %%
df_train = pd.read_csv(dataset_dir +"/df_train.csv")
df_test = pd.read_csv(dataset_dir +"/df_test.csv")

train_input = pickle.load(open(pickle_dir+"/train_input.pickle", "rb"))
test_input = pickle.load(open(pickle_dir+"/test_input.pickle", "rb"))
train_label = pickle.load(open(pickle_dir+"/train_label.pickle", "rb"))
test_label = pickle.load(open(pickle_dir+"/test_label.pickle", "rb"))
df_train_shapley = pd.read_csv(dataset_dir +"/df_train_shapley.csv")
df_test_shapley = pd.read_csv(dataset_dir +"/df_test_shapley.csv")

df_train_topic_tag = pickle.load(open(pickle_dir+"/topic_tag.pickle", "rb"))

# %%
topics = 20
my_list = [str(i) for i in np.arange(topics)]
string = 'topic '
input_columns = list(map(lambda orig_string: string + orig_string, my_list))

df_train_input = pd.DataFrame(train_input, columns = input_columns)
df_train_label = pd.DataFrame(train_label, columns = ['label'])

df_test_input = pd.DataFrame(test_input, columns = input_columns)
df_test_label = pd.DataFrame(test_label, columns = ['label'])


# %%
def prep_df(df_shapley, df_input):
    df_2 = df_shapley.copy().iloc[:, :20]
    columns_dict = {0: "topic 0", 1: "topic 1", 
                2: "topic 2", 3: "topic 3",
                4: "topic 4", 5: "topic 5",
                6: "topic 6", 7: "topic 7",
                8: "topic 8", 9: "topic 9",
                10: "topic 10", 11: "topic 11", 
                12: "topic 12", 13: "topic 13",
                14: "topic 14", 15: "topic 15",
                16: "topic 16", 17: "topic 17",
                18: "topic 18", 19: "topic 19",
                }
    
    pd_idx = pd.Index(list(range(20)))
    boolean_idx_matrix = np.zeros((len(df_shapley), 20), dtype=bool)
    topic_list = list(df_input.apply(lambda x: x > 0, raw=True).apply(lambda x: list(df_input.columns[x.values]), axis=1))

    for idx in range(len(df_2)):
        boolean_idx_matrix[idx] = pd_idx.isin([k for k, v in columns_dict.items() if v in topic_list[idx]])

    for i in range(boolean_idx_matrix.shape[0]):
        for j in range(boolean_idx_matrix.shape[1]):
            if (boolean_idx_matrix[i][j] == False):
                df_2.at[i, columns_dict[j]] = 0
    
    df_3 = df_2.abs()
    df_4 = df_3.div(df_3.sum(axis=1), axis=0) 

    return df_2, df_3, df_4


# %%
train_df2, train_df3, train_df4 = prep_df(df_train_shapley, df_train_input)
test_df2, test_df3, test_df4 = prep_df(df_test_shapley, df_test_input)

# %%
indexes_public = list(df_test[df_test['label'] == 1.0].index)
indexes_private = list(df_test[df_test['label'] != 1.0].index)
len(indexes_public), len(indexes_private)
# %% [markdown]
# # Dominant category

# %%
def cat_dominant(dominant_private_ub, dominant_public_ub, df_4, label):
    
    print("***************************DOMINANT****************************")
    if label:
        df_dominant_public_base = df_4.loc[[0]]
        print(df_dominant_public_base.max(axis=1) )

        df_dominant_public = df_dominant_public_base[df_dominant_public_base.max(axis=1) >= dominant_public_ub]
        return df_dominant_public
    else:
        df_dominant_private_base = df_4.loc[[0]]
        df_dominant_private = df_dominant_private_base[df_dominant_private_base.max(axis=1) >= dominant_private_ub]
        print(df_dominant_private)
        return df_dominant_private



# %% [markdown]
# # Opponent category

# Define the cat_opponent function
def cat_opponent(opponent_private_ub, opponent_public_ub, df_4, df_2, opponent_ub_2, df_dominant, paired_ub, label):
    # Convert the numpy arrays to DataFrames
    print("***************************OPPPOPNNENET****************************")
    if label:
        df_5_public_base = df_4.loc[[0]]
        df_5_public = df_5_public_base.apply(lambda x: x >= opponent_public_ub)
        df_5 = df_5_public
    else:
        df_5_private_base = df_4.loc[[0]]
        df_5_private = df_5_private_base.apply(lambda x: x >= opponent_private_ub)

        df_5 = df_5_private 

    df_6 = df_5.loc[df_5[(df_5.sum(axis=1) >= 2)].index]
    df_6 = df_2[df_6].dropna(how='all')

    df_n_n = df_6[df_6.apply(lambda x: x < 0)] < 0
    df_n_p = df_6[df_6.apply(lambda x: x > 0)] > 0
    df_n = df_n_n.sum(axis=1).to_frame('negatives')
    df_n["positives"] = df_n_p.sum(axis=1)

    df_opponent = df_n[(df_n["negatives"] >= opponent_ub_2) & (df_n["positives"] >= opponent_ub_2)]
    idx_intersect = list(set(df_opponent.index) & set(df_dominant.index))
    df_opponent = df_opponent.drop(idx_intersect)
    df_opponent = df_4.iloc[df_opponent.index]

    temp_idx_list = [idx if (np.abs(df_2.iloc[idx].min() + df_2.iloc[idx].max()) < paired_ub) else None for idx in df_opponent.index]
    indexes_filtered_opponent = [x for x in temp_idx_list if x is not None]

    return indexes_filtered_opponent, df_opponent


# Define the cat_collab function
def cat_collab(df_2, collaborative_private_ub, collaborative_public_ub, df_dominant, df_conflict, label):
    # Convert the numpy arrays to DataFrames
    
    print("***************************Collababababaa****************************")
    df_7 = df_2.copy()
    df_8 = df_7[df_7.apply(lambda x: x < 0)].sum(axis=1).to_frame('negatives')
    df_8["positives"] = df_7[df_7.apply(lambda x: x > 0)].sum(axis=1)
    df_8 = df_8.abs()
    df_8["summa"] = df_8.sum(axis=1)
    df_8["res_neg"] = df_8["negatives"] / df_8["summa"]
    df_8["res_pos"] = df_8["positives"] / df_8["summa"]

    if label:
        df_8_public_base = df_8.loc[[0]]
        l_neg_public = list(df_8_public_base[df_8_public_base["res_neg"] >= collaborative_public_ub].index)
        l_pos_public = list(df_8_public_base[df_8_public_base["res_pos"] >= collaborative_public_ub].index)
        df_collaborative_public = df_7.loc[[*l_neg_public, *l_pos_public]].sort_index()
        df_collaborative = df_collaborative_public
    else:
        df_8_private_base = df_8.loc[[0]]
        l_neg_private = list(df_8_private_base[df_8_private_base["res_neg"] >= collaborative_private_ub].index)
        l_pos_private = list(df_8_private_base[df_8_private_base["res_pos"] >= collaborative_private_ub].index)
        df_collaborative_private = df_7.loc[[*l_neg_private, *l_pos_private]].sort_index()
        df_collaborative = df_collaborative_private
  
    idx_intersect_dom = list(set(df_collaborative.index) & set(df_dominant.index))
    idx_intersect_con = list(set(df_collaborative.index) & set(df_conflict.index))
    df_collaborative = df_collaborative.drop(idx_intersect_dom + idx_intersect_con)

    return df_collaborative

def dict_top(test_df_dominant, test_df_opponent,test_df_collaborative, test_df_weak):
    dominant_dict = {}
    opposing_dict = {}
    weak_dict = {}
    collaborative_dict = {}
    top_weak = pd.DataFrame()
    if not test_df_dominant.empty:
        dominant_dict = test_df_dominant.idxmax(axis=1).to_dict()
    if not test_df_opponent.empty:
        opposing_dict = {index: (value1, value2) for index, value1, value2 in zip(test_df2.loc[list(test_df_opponent.index)].idxmin(axis=1).index, test_df2.loc[list(test_df_opponent.index)].idxmin(axis=1).values, test_df2.loc[list(test_df_opponent.index)].idxmax(axis=1).values)}
    if not test_df_collaborative.empty:
        df = test_df4.loc[list(test_df_collaborative.index)]
        collaborative_dict = df.apply(lambda row: row.nlargest(3).index.tolist(), axis=1).to_dict()
    if not test_df_weak.empty:
        top_weak = test_df4.loc[test_df_weak.index].apply(lambda row: row.nlargest(3).index.tolist(), axis=1)

    top_negative = test_df_weak.apply(lambda row: row.nlargest(3).index.tolist(), axis=1)
    top_positive = test_df_weak.apply(lambda row: row.nsmallest(3).index.tolist(), axis=1)

    intersection_negative = {}
    intersection_positive = {}
    for idx in top_weak.index:
        print("********************TOP_WEAK_INDEX********************")
        print(idx)
        intersection_neg = set(top_weak[idx]).intersection(top_negative[idx])
        intersection_pos = set(top_weak[idx]).intersection(top_positive[idx])

        intersection_negative[idx] = list(intersection_neg)
        intersection_positive[idx] = list(intersection_pos)


    weak_dict = {key: [intersection_negative.get(key, []), intersection_positive.get(key, [])] for key in set(intersection_negative) | set(intersection_positive)}
    return dominant_dict, opposing_dict, collaborative_dict, weak_dict

def merge_dict_category_topic(dominant_dict, opposing_dict, collaborative_dict, weak_dict):
    cat_top_dict = {}

    for key, values in dominant_dict.items():
        cat_top_dict[key] = ['dominant', values]

    for key, values in opposing_dict.items():
        cat_top_dict[key] = ["opposing", values]

    for key, values in collaborative_dict.items():
        cat_top_dict[key] = ['collaborative', values]

    for key, values in weak_dict.items():
        cat_top_dict[key] = ["weak", values]

    return dict(sorted(cat_top_dict.items(), key=lambda item: item[0]))

def plot_explanations(idx, cat_top_dict, label, w_comma):
    category_name = cat_top_dict.get(idx)[0]
    label = "private" if label == 0 else "public" 
    print("*******************CAT_TOP_DICT*********************************")
    print(cat_top_dict)
    truth_label = label
    # image_id = 4398397540
    predicted_label = label

    # Function to format text
    def format_text(text, max_width):
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            if len(' '.join(current_line + [word])) <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)
    topics_dict = {"topic 0":"Seaside", "topic 1": "Competition", "topic 2": "Urban", "topic 3":"Room",
                  "topic 4":"Animal", "topic 5":"Nature", "topic 6":"Business", "topic 7": "Offense", "topic 8":"Performance",
                  "topic 9":"Studio", "topic 10":"People", "topic 11":"Garden", "topic 12":"Road", "topic 13":"Child", "topic 14":"Group", 
                  "topic 15":"Politics", "topic 16":"Design", "topic 17":"Sky", "topic 18": "Snow", "topic 19": "Fashion"}
    # Generate text based on category
    if category_name == "dominant":
        topic = cat_top_dict.get(idx)[1]
        topic_text = topics_dict[topic]
        # feel free to rename topics based on your topic modelling result
        text = f"The generated explanation for this image being assigned to the {predicted_label} class \
                is that it is related to the topic {topic_text} with these specific tags."

        word_cloud_topics = [topic]
        num_circles = len(word_cloud_topics)

        if truth_label == predicted_label and truth_label == "private":
            contour_colors = ["darkviolet"]
        elif truth_label == predicted_label and truth_label == "public":
            contour_colors = ["darkorange"]
        else: # misclassification
            contour_colors = ["black"]

    elif category_name == "opposing":
        topic_negative = cat_top_dict.get(idx)[1][0]
        topic_negative_text = topics_dict[topic_negative]
        topic_positive = cat_top_dict.get(idx)[1][1]
        topic_positive_text = topics_dict[topic_positive]
        # feel free to rename topics based on your topic modelling result
        text = f"Even though the image is related to the topic {topic_negative_text} with the specific tags below \
                (which signals the {truth_label} class), it is also related to the topic {topic_positive_text} and \
                for that reason, it is classified as {predicted_label}."

        word_cloud_topics = [topic_positive, topic_negative]
        num_circles = len(word_cloud_topics)
        print("opposing", num_circles)
        if truth_label == predicted_label and truth_label == "private":
            contour_colors = ["darkviolet", "darkorange"]
        elif truth_label == predicted_label and truth_label == "public":
            contour_colors = ["darkorange", "darkviolet"]
        else: # misclassification
            contour_colors = ["black", "black"]

    elif category_name == "collaborative":
        topics = cat_top_dict.get(idx)[1]
        topics_text = []
        for topic in topics:
            topics_text.append(topics_dict[topic])
        # feel free to rename topics based on your topic modelling result
        text = f"The generated explanation for this image being assigned to the {predicted_label} class \
                is that it is related to the topics {', '.join(topics_text)} with these specific tags."

        word_cloud_topics = topics
        num_circles = len(word_cloud_topics)
        print("collab", num_circles)

        if truth_label == predicted_label and truth_label == "private":
            contour_colors = ["darkviolet", "darkviolet", "darkviolet"]
        elif truth_label == predicted_label and truth_label == "public":
            contour_colors = ["darkorange", "darkorange", "darkorange"]
        else: # misclassification
            contour_colors = ["black", "black", "black"]

    else:
        topic_negative = cat_top_dict.get(idx)[1][0]
        topic_positive = cat_top_dict.get(idx)[1][1]
        topic_negative_text = []
        for topic in topic_negative:
            topic_negative_text.append(topics_dict[topic])
        topic_positive_text = []
        for topic in topic_positive:  
            topic_positive_text.append(topics_dict[topic])
        # feel free to rename topics based on your topic modelling result
        text = f"Even though the image is related to the topic {', '.join(topic_negative_text)} with the specific tags below \
                which signals the {truth_label} class), it is also related to the topic {', '.join(topic_positive_text)} and \
                for that reason, it is classified as {predicted_label}."

        word_cloud_topics = [topic_positive, topic_negative]
        num_circles = len(word_cloud_topics)

        if len(topic_positive)==1 and truth_label == predicted_label and truth_label == "private":
            contour_colors = ["darkviolet", "darkorange", "darkorange"]
        elif len(topic_positive)==2 and truth_label == predicted_label and truth_label == "private":
            contour_colors = ["darkviolet", "darkviolet", "darkorange"]
        elif len(topic_positive)==3 and truth_label == predicted_label and truth_label == "private":
            contour_colors = ["darkviolet", "darkviolet", "darkviolet"]
        else: # misclassification
            contour_colors = ["black", "black", "black"]

    formatted_text = format_text(text, max_width=80)

    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)

    contour_color = "darkviolet"  # or darkorange

    # Calculate the number of columns based on the number of circles
    num_circles = len(word_cloud_topics)
    # Create the subplots grid
    num_cols = num_circles + 1  # One extra column for the text
    fig, axs = plt.subplots(1, num_cols, figsize=(5 * num_cols, 10))

    # Plot the text at the top
    axs[0].text(0.5, 0.5, formatted_text, ha='center', va='center', fontsize=12)
    axs[0].axis("off")

    # Create WordClouds and plot circles in the remaining columns
    wordclouds = []  # List to hold the generated WordCloud objects

    for i in range(num_circles):
        wordcloud = WordCloud(width=500, height=300, margin=3, prefer_horizontal=0.7, scale=1,
                              background_color="white", mask=mask, contour_width=0.1,
                              contour_color=contour_colors[i], relative_scaling=0)

        x = w_comma
        print(x)
        print("**********************WORD_CLOUD_TOPICS[i]*********************************")
        print(word_cloud_topics[i])
        if type(word_cloud_topics[i])== list:

            topic_id = int(word_cloud_topics[i][0].split()[-1])
        elif type(word_cloud_topics) == str:
            topic_id = int(word_cloud_topics.split()[-1])
        else:
            topic_id = int(word_cloud_topics[i].split()[-1])
        print(list(df_train_topic_tag[topic_id]))
        tags = sorted(set(list(df_train_topic_tag[topic_id])) & set(x), key = list(df_train_topic_tag[topic_id]).index)
        tags = " ".join(tags)
        print(tags)
        if len(tags) == 0:
            continue
        wordcloud.generate(tags)

        axs[i + 1].imshow(wordcloud)
        axs[i + 1].title.set_text(word_cloud_topics[i])
        axs[i + 1].axis("off")
        fig = plt.figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)

        # Plot the WordCloud onto the canvas
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(wordcloud)
        ax.axis("off")

        # Convert the figure to a PNG image in memory
        buffer = io.BytesIO()
        canvas.print_png(buffer)

        # Encode the image data as base64
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        if type(word_cloud_topics[i]) == list:
            wordclouds.append((image_data, topics_dict[word_cloud_topics[i][0]]))
        elif type(word_cloud_topics)== str:
            wordclouds.append((image_data, topics_dict[word_cloud_topics]))
        else:
            wordclouds.append((image_data, topics_dict[word_cloud_topics[i]]))
    return wordclouds, text

# transform extracted_tags column into comma separated str list
def str_to_comma_list_single_instance(extracted_tags):
    extracted_tags2 = [i.split(':', 1)[0] for i in extracted_tags][:20]

    tag_list = [x.lower().strip().replace('(', '').replace(')', '').replace('-', ' ') for x in extracted_tags2]
    tag_list = [x.replace(' ', '_') for x in tag_list]
    cleaned_tags_w_comma = ','.join(tag_list)
    return cleaned_tags_w_comma


def tag_intersect_list(image_id, topic_id, df, df_train_topic_tag):
    x = df[df.image == image_id].cleaned_tags_w_comma.item()

    intersect_list = sorted(set(list(df_train_topic_tag[topic_id])) & set(ast.literal_eval(x)), key = list(df_train_topic_tag[topic_id]).index)
    return intersect_list


def generate_exp(photo_topics, label, tags):
    exp = show_topics_contributions(photo_topics, label)
    print("***********************EXP****************************")
    print(exp)
    exp = exp.values

    topics = 20
    my_list = [str(i) for i in np.arange(topics)]
    input_columns = list(map(lambda orig_string: 'topic ' + orig_string, my_list))
    df_exp = pd.DataFrame(exp, columns = input_columns)
    print("******************************DF_EXP******************************")
    print(df_exp)   
    df_2, df_3, df_4 = prep_df(df_exp, photo_topics)
    cleaned_tags_w_comma = str_to_comma_list_single_instance(tags)
    df_dominant = cat_dominant(0.7, 0.7, df_4, label)
    print(df_dominant)
    df_opponent = pd.DataFrame()
    df_collaborative = pd.DataFrame()
    df_weak = pd.DataFrame()
    if not df_dominant.empty:
        name = "dominant"
        dominant_dict, opposing_dict, colloborative_dict, weak_dict = dict_top(df_dominant, df_opponent,df_collaborative, df_weak)
        cat_top_dict = merge_dict_category_topic(dominant_dict, opposing_dict, colloborative_dict, weak_dict)
        print("*********************************CAT_TOP_DICT*****************************")
        print(cat_top_dict)
        wordcloud, text = plot_explanations(0, cat_top_dict, label, tags)
        return df_dominant, name, wordcloud, text
    indexes_opponent, df_opponent = cat_opponent(0.2, 0.2, df_4, df_2, 1, df_dominant, 0.1, label)
    print(df_opponent)
    if not df_opponent.empty:
        dominant_dict, opposing_dict, colloborative_dict, weak_dict = dict_top(df_dominant, df_opponent,df_collaborative, df_weak)
        cat_top_dict = merge_dict_category_topic(dominant_dict, opposing_dict, colloborative_dict, weak_dict)
        print("*********************************CAT_TOP_DICT*****************************")
        print(cat_top_dict)
        wordcloud, text = plot_explanations(0, cat_top_dict, label, tags)
        name = "opponent"
        return df_opponent, name, wordcloud, text

    df_collaborative = cat_collab(df_2, 0.8, 0.8, df_dominant, df_opponent, label)
    print(df_collaborative)
    if not df_collaborative.empty:
        name = "collaborative"
        dominant_dict, opposing_dict, colloborative_dict, weak_dict = dict_top(df_dominant, df_opponent,df_collaborative, df_weak)
        cat_top_dict = merge_dict_category_topic(dominant_dict, opposing_dict, colloborative_dict, weak_dict)
        print("*********************************CAT_TOP_DICT*****************************")
        print(cat_top_dict)
        wordcloud, text = plot_explanations(0, cat_top_dict, label, tags)

        return df_collaborative, name, wordcloud, text
    
    else:
        df_weak = df_exp[~df_exp.index.isin(list(df_dominant.index)+list(df_opponent.index)+list(df_collaborative.index))]
        print(df_weak)
        name = "weak"
        dominant_dict, opposing_dict, colloborative_dict, weak_dict = dict_top(df_dominant, df_opponent,df_collaborative, df_weak)
        cat_top_dict = merge_dict_category_topic(dominant_dict, opposing_dict, colloborative_dict, weak_dict)
        print("*********************************CAT_TOP_DICT*****************************")
        print(cat_top_dict)
        wordcloud, text = plot_explanations(0, cat_top_dict, label, tags)

        return df_weak, name, wordcloud, text

# %% [markdown]
# # Observing misclassified images

# %%
