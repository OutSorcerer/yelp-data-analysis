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

import feature_extractor

# +
# Extract deep features for every photo (it takes approximately 3 hours on my hardware) or load them from cache.
photos_with_features_df_full = feature_extractor.load_photos_with_features_df()

# Use a random subsample of 2k photos, as 200k are too much to feed into t-SNE and visualize.
photos_with_features_df = photos_with_features_df_full.sample(n=2000, random_state=42).reset_index()
# -

# Stack all features in a singe array of shape (num_photos, num_features)
features_array = np.stack(photos_with_features_df.features.values, axis=0)

# +
# Let's apply t-SNE to visualize our features.
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

# +
# Now let's try to visualize the results of 3d t-SNE.

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

# +
# How well can we recover photo labels using an unsupervised algoritm?
# Try to cluster everything into 5 clusters and then compute NMI score between known clusering of photos by labels.
true_labels = LabelEncoder().fit_transform(photos_with_features_df.label)
print(true_labels)

clustering_result = AgglomerativeClustering(n_clusters=5).fit_predict(features_array)
print(clustering_result)

nmi = normalized_mutual_info_score(true_labels, clustering_result)
print(nmi)

# For comparison NMI with a random clustering will give much lower score.
random_labels = np.random.randint(low=0, high=4, size=true_labels.shape)
nmi_random = normalized_mutual_info_score(true_labels, random_labels)
print(nmi_random)

# And for perfect match it returns 1.
nmi_perfect = normalized_mutual_info_score(true_labels, true_labels)
print(nmi_perfect)
# -


# So, the result is better than random, but is not so good.
#
# Maybe, because clusters are not mutually exclusive (some "food" photos have backround and some "inside" photos can have food on them) ? 

# +
# How well can we recover photo labels using a supervised algorithm?


# -


