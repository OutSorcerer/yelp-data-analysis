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

import feature_extractor
import seaborn as sns
# -

# Extract deep features for every photo (it takes approximately 3 hours on my hardware) or load them from cache.
photos_with_features_df = feature_extractor.load_photos_with_features_df()

# Stack all features in a singe array of shape (num_photos, num_features)
features_array = np.stack(photos_with_features_df.features.values, axis=0)

# +
# Let's apply t-SNE to visualize our features.
# For each photo we know its label ('inside', 'food', 'outside', 'drink').
# Let's see how they are arranged after embedding feaures into 2d scape with t-SNE.
# t-SNE has no information about labels, they are added just for visualization. 

tsne_2d = TSNE(n_components=2, verbose=1, n_iter=300)
tsne_features_2d = tsne_2d.fit_transform(features_array)

tsne_2d_df = pd.DataFrame()
tsne_2d_df['label'] = photos_with_features_df['label']
tsne_2d_df['tsne-2d-one'] = tsne_features_2d[:, 0]
tsne_2d_df['tsne-2d-two'] = tsne_features_2d[:, 1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    palette=sns.xkcd_palette(["blue", "red", "green", "yellow"]),
    data=tsne_2d_df,
    legend="full"    
)


# +
# Now let's try to use 3d t-SNE instead of 2d.

tsne_3d = TSNE(n_components=3, verbose=1, n_iter=400)
tsne_features_3d = tsne_3d.fit_transform(features_array)

label_to_color = {'inside':'b',
                  'food':'r',
                  'outside':'g',
                  'drink':'y'}

inside_indices = photos_with_features_df[photos_with_features_df.label=='inside'].index
food_indices = photos_with_features_df[photos_with_features_df.label=='food'].index
outside_indices = photos_with_features_df[photos_with_features_df.label=='outside'].index
drink_indices = photos_with_features_df[photos_with_features_df.label=='drink'].index

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
    
ax.set_xlabel('tsne-3d-one')
ax.set_ylabel('tsne-3d-two')
ax.set_zlabel('tsne-3d-three')
plt.legend()
plt.show()
