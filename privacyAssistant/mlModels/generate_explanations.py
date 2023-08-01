# %%


# %%
import pandas as pd
import numpy as np
import pickle
from IPython.display import display
import shap
from .classify_images import show_topics_contributions

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
    print("********************newdf22222222222222222222222222222")
    print(df_2)
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
    print(df_4)
    if label:
        df_dominant_public_base = df_4.loc[[0]]
        print("****************df_dominant_public_base.max(axis=1) ************************")
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
    print(df_4)
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
    print(df_2)
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


def generate_exp(photo_topics, label):
    exp = show_topics_contributions(photo_topics, label).values

    topics = 20
    my_list = [str(i) for i in np.arange(topics)]
    input_columns = list(map(lambda orig_string: 'topic ' + orig_string, my_list))
    df_exp = pd.DataFrame(exp, columns = input_columns)
    print(df_exp)   
    df_2, df_3, df_4 = prep_df(df_exp, photo_topics)
    
    df_dominant = cat_dominant(0.7, 0.7, df_4, label)
    if not df_dominant.empty:
        name = "dominant"
        return df_dominant, name
    indexes_opponent, df_opponent = cat_opponent(0.2, 0.2, df_4, df_2, 1, df_dominant, 0.1, label)
    if not df_opponent.empty:
        name = "dominant"
        return df_opponent, name

    df_collaborative = cat_collab(df_2, 0.8, 0.8, df_dominant, df_opponent, label)
    if not df_collaborative.empty:
        name = "dominant"
        return df_collaborative, name
    
    else:
        df_weak = df_exp[~df_exp.index.isin(list(df_dominant.index)+list(df_opponent.index)+list(df_collaborative.index))]
        name = "weak"
        return df_weak, name

# %% [markdown]
# # Observing misclassified images

# %%
