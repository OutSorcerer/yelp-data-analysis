# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # What information can we get from photos of businesses?
#  
# Working with raw image data is infeasible, not only because its total approximate size is `200k*400*400*3 ≈ 100GB`.
#
# Also, direct pixel values are not very useful as features as photos of same objects can be quite far away in RGB space.
#
# So, let us use deep features.
#
# Following [CNN features are also great at unsupervised classification](https://arxiv.org/abs/1707.01700) that demonstrates that combination of Xception architecture and Agglomerative Hierarchical Clustering gives the best clustering results (using Normalized Mutual Information score) let us start with these.
#
# The implementation of Xception trained on ImageNet is available at https://github.com/Cadene/pretrained-models.pytorch.
#
# We need to get features after the last pooling layer so `xception_with_pooling_features.py` was writen to achive this.

# +
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import feature_extractor

# +
# Extract deep features for every photo (it takes approximately 3 hours on my hardware) or load them from cache.
photos_with_features_df_full = feature_extractor.load_photos_with_features_df()

# Use a random subsample of 20k photos for training a logistic regression.
photos_with_features_df_20k = photos_with_features_df_full.sample(n=20000, random_state=42).reset_index()

# Use a random subsample of 2k photos, as 200k are too much to feed into t-SNE and visualize.
photos_with_features_df = photos_with_features_df_full.sample(n=2000, random_state=42).reset_index()

# +
# Stack all features in a singe array of shape (num_photos, num_features)
features_array_full = np.stack(photos_with_features_df_full.features.values, axis=0)

# Do the same for a 20k subsample
features_array_20k = np.stack(photos_with_features_df_20k.features.values, axis=0)

# Do the same for a 2k subsample
features_array = np.stack(photos_with_features_df.features.values, axis=0)
# -

# Transorm string labels to class indices
true_labels_full = LabelEncoder().fit_transform(photos_with_features_df_full.label)
true_labels_20k = LabelEncoder().fit_transform(photos_with_features_df_20k.label)
true_labels = LabelEncoder().fit_transform(photos_with_features_df.label)

# ## Let's apply 2d t-SNE to visualize our features.

# +
# For each photo we know its label ('inside', 'food', 'outside', 'drink', 'menu').
# Let's see how they are arranged after embedding feaures into 2d scape with t-SNE.
# t-SNE has no information about labels, they are added just for visualization. 

tsne_2d = TSNE(n_components=2, verbose=1, n_iter=1000, perplexity=10, random_state=42)
tsne_features_2d = tsne_2d.fit_transform(features_array)

tsne_2d_df = pd.DataFrame()
tsne_2d_df['label'] = photos_with_features_df['label']
tsne_2d_df['tsne-2d-one'] = tsne_features_2d[:, 0]
tsne_2d_df['tsne-2d-two'] = tsne_features_2d[:, 1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    palette=sns.xkcd_palette(["blue", "red", "green", "yellow", 'magenta']),
    data=tsne_2d_df,
    legend="full"    
)
# -


# We see some clumps, that could, for example, correspond to certain types of food, like burgers or pizza, however [it is really easy to misinterpret t-SNE results](https://distill.pub/2016/misread-tsne/).

# ## Now let's try to visualize the results of 3d t-SNE

# +
tsne_3d = TSNE(n_components=3, verbose=1, n_iter=1000, perplexity=5, random_state=42)
tsne_features_3d = tsne_3d.fit_transform(features_array)

label_to_color = {'inside':'blue',
                  'food':'red',
                  'outside':'green',
                  'drink':'yellow',
                  'menu': 'magenta',
                 }

inside_indices = photos_with_features_df[photos_with_features_df.label=='inside'].index
food_indices = photos_with_features_df[photos_with_features_df.label=='food'].index
outside_indices = photos_with_features_df[photos_with_features_df.label=='outside'].index
drink_indices = photos_with_features_df[photos_with_features_df.label=='drink'].index
menu_indices = photos_with_features_df[photos_with_features_df.label=='menu'].index

ax = plt.figure(figsize=(16,10)).gca(projection='3d')

def plot_label(indices, label):
    ax.scatter(
        xs=tsne_features_3d[indices, 0], 
        ys=tsne_features_3d[indices, 1], 
        zs=tsne_features_3d[indices, 2], 
        c=photos_with_features_df.loc[indices, 'label'].map(lambda label: label_to_color[label]),
        label=label
    )

plot_label(inside_indices, 'inside')
plot_label(food_indices, 'food')
plot_label(outside_indices, 'outside')
plot_label(drink_indices, 'drink')
plot_label(menu_indices, 'menu')
    
ax.set_xlabel('tsne-3d-one')
ax.set_ylabel('tsne-3d-two')
ax.set_zlabel('tsne-3d-three')
plt.legend()
plt.show()
# -

# Some labels are almost invisible now since they ended up somewhere inside the point cloud.

# ## How well can we recover photo labels using an unsupervised algoritm?

# +
# Try to cluster everything into 5 clusters and then compute NMI score between known clusering of photos by labels.
print('True labels:       {}'.format(true_labels))

clustering_result = AgglomerativeClustering(n_clusters=5).fit_predict(features_array)
print('Clustering result: {}'.format(clustering_result))

nmi = normalized_mutual_info_score(true_labels, clustering_result, average_method='arithmetic')
print('NMI={}'.format(nmi))

# For comparison NMI with a random clustering will give much lower score.
np.random.seed(42)
random_labels = np.random.randint(low=0, high=4, size=true_labels.shape)
nmi_random = normalized_mutual_info_score(true_labels, random_labels, average_method='arithmetic')
print('random NMI={}'.format(nmi_random))

# And for perfect match it returns 1.
nmi_perfect = normalized_mutual_info_score(true_labels, true_labels, average_method='arithmetic')
print('perfect NMI={}'.format(nmi_perfect))
# -


# So, the result is better than random, but is not so good.
#
# Maybe, because clusters are not mutually exclusive (some "food" photos have backround and some "inside" photos can have food on them)? 

# ## How well can we recover photo labels using a supervised algorithm?

# Split data into train and test datasets for validation.
X_train, X_test, y_train, y_test = train_test_split(features_array_20k, true_labels_20k, test_size=0.2, random_state=42)
# + {}
classifier = Pipeline([
    ('scaler', StandardScaler()),
    ('logit', LogisticRegression(solver='lbfgs', C=0.01, verbose=0, max_iter=2000, multi_class='multinomial',
                                 class_weight='balanced', random_state=42))])

# Classes are imbalanced, let's see their frequencies
y_frequencies = y_counts / np.sum(y_counts)
print('class frequencies: {}'.format(y_frequencies))

# As a baseline compute metrics for a trivial classifier that always outputs the most popular class.
y_counts = np.bincount(y_train)
most_frequent_y = np.full(shape=y_test.shape, fill_value=np.argmax(y_counts)) # Food
print('Mean constant accuracy: {}'.format(accuracy_score(y_test, most_frequent_y)))
print('Balanced mean constant accuracy: {}'.format(balanced_accuracy_score(y_test, most_frequent_y)))

# and for a classifier that outputs a random class.
np.random.seed(42)
random_y = np.random.choice(y_counts.shape[0], size=y_test.shape, p=y_frequencies)
print('Mean random accuracy: {}'.format(accuracy_score(y_test, random_y)))
print('Balanced mean random accuracy: {}'.format(balanced_accuracy_score(y_test, random_y)))

classifier = classifier.fit(X_train, y_train)
y_train_pred = classifier.predict(X_train)
print('Mean train accuracy: {}'.format(accuracy_score(y_train, y_train_pred)))
print('Balanced mean train accuracy: {}'.format(balanced_accuracy_score(y_train, y_train_pred)))

y_test_pred = classifier.predict(X_test)
print('Mean test accuracy: {}'.format(accuracy_score(y_test, y_test_pred)))
print('Balanced mean test accuracy: {}'.format(balanced_accuracy_score(y_test, y_test_pred)))

# -


# Even though there is overfitting, metrics are much better comparing to constant and random predictions.

# +
# What about NMI of a supervised algorithm vs. real labels?

nmi_supervised = normalized_mutual_info_score(y_test, y_test_pred, average_method='arithmetic')
print('supervised NMI={}'.format(nmi_supervised))

# -

# NMI with a supervised algorithm is much better.
