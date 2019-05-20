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
# Also, direct pixel values are not very useful as features of photos of same objects can be quite far away in RGB space.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

import feature_extractor
# -

# Load businesses
business_df = pd.read_json('dataset/business.json', lines=True)

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

# ## How well can we recover photo labels using an unsupervised algorithm?

# +
# Try to cluster everything into 5 clusters and then compute NMI score between known clustering of photos by labels.
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
y_counts = np.bincount(y_train)
y_frequencies = y_counts / np.sum(y_counts)
print('class frequencies: {}'.format(y_frequencies))

# As a baseline compute metrics for a trivial classifier that always outputs the most popular class.
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

# ### What about NMI of a supervised algorithm vs. real labels?

nmi_supervised = normalized_mutual_info_score(y_test, y_test_pred, average_method='arithmetic')
print('supervised NMI={}'.format(nmi_supervised))


# NMI with a supervised algorithm is much better.

# ## How well can we recover business attributes by its photo features?

# ### Let's take GoodForKids as an example

# +
# Get good_for_kids feature for all businesses and add it into business_df
good_for_kids = business_df.attributes.apply(lambda json: json['GoodForKids'] == 'True' if json is not None and 'GoodForKids' in json else None)
business_df['good_for_kids'] = good_for_kids

print('good_for_kids stats by business:')
good_for_kids.value_counts(dropna=False)

# +
# Let's merge photos data frame with businesses data frame, 
# but only for businesses where we exactly know if they are good for kids.
business_is_good_for_kids_df = business_df[['business_id', 'good_for_kids']]
business_is_good_for_kids_df = business_is_good_for_kids_df[~business_is_good_for_kids_df.good_for_kids.isnull()]
photo_is_good_for_kids_df = pd.merge(left=photos_with_features_df_full, right=business_is_good_for_kids_df, how='inner', on='business_id')
print('good_for_kids stats by photo:')
print(photo_is_good_for_kids_df.good_for_kids.value_counts(dropna=False))

# Sample 20k of examples and get Xs and ys.
photo_is_good_for_kids_df = photo_is_good_for_kids_df.sample(n=20000, random_state=42).reset_index()
photo_is_good_for_kids_features = np.stack(photo_is_good_for_kids_df.features.values, axis=0)
photo_is_good_for_kids_y = photo_is_good_for_kids_df.good_for_kids.values.astype(np.bool)
photo_is_good_for_kids_y_business_id = photo_is_good_for_kids_df.business_id.values
# -

# Even though most businesses are not good for kids, most photos belong to places that are good for kids.

# Split good_for_kids data into train and test datasets for validation.
X_kids_train, X_kids_test, y_kids_train, y_kids_test, business_id_train, business_id_test = train_test_split(photo_is_good_for_kids_features, photo_is_good_for_kids_y, photo_is_good_for_kids_y_business_id, test_size=0.2, random_state=42)

# +
classifier_kids = Pipeline([
    ('scaler', StandardScaler()),
    ('logit', LogisticRegression(solver='lbfgs', C=0.0001, verbose=0, max_iter=2000, 
                                class_weight='balanced', random_state=42))    
    ])

# What percentage of photos belong to businesses that are good for kids?
y_kids_counts = np.bincount(y_kids_train)
y_kids_frequencies = y_kids_counts / np.sum(y_kids_counts)
print('class frequencies: {}'.format(y_kids_frequencies))

# As a baseline compute metrics for a trivial classifier that always outputs the most popular class.
most_frequent_y_kids = np.full(shape=y_kids_test.shape, fill_value=bool(np.argmax(y_kids_counts)))
print('Mean constant accuracy: {}'.format(accuracy_score(y_kids_test, most_frequent_y_kids)))
print('Balanced mean constant accuracy: {}'.format(balanced_accuracy_score(y_kids_test, most_frequent_y_kids)))

# and for a classifier that outputs a random class.
np.random.seed(42)
random_y_kids = np.random.choice([False, True], size=y_kids_test.shape, p=y_kids_frequencies)
print('Mean random accuracy: {}'.format(accuracy_score(y_kids_test, random_y_kids)))
print('Balanced mean random accuracy: {}'.format(balanced_accuracy_score(y_kids_test, random_y_kids)))

classifier_kids = classifier_kids.fit(X_kids_train, y_kids_train)
y_kids_train_pred = classifier_kids.predict(X_kids_train)
print('Mean train accuracy: {}'.format(accuracy_score(y_kids_train, y_kids_train_pred)))
print('Balanced mean train accuracy: {}'.format(balanced_accuracy_score(y_kids_train, y_kids_train_pred)))

y_kids_test_pred = classifier_kids.predict(X_kids_test)
y_kids_test_pred_proba = classifier_kids.predict_proba(X_kids_test)[:, 1]
print('Mean test accuracy: {}'.format(accuracy_score(y_kids_test, y_kids_test_pred)))
print('Balanced mean test accuracy: {}'.format(balanced_accuracy_score(y_kids_test, y_kids_test_pred)))
print('Test average precision: {}'.format(average_precision_score(y_kids_test, y_kids_test_pred_proba)))
# -

# So, given a photo we can try to predict if a photo belongs to a business that is good for kids,
# but with a quality that is far from perfect (even though balanced accuracy is better that constant prediction, unbalanced accuracy is even worse than constant prediction).

# #### What if we aggregate predictions for all photos that belong to the same business?

# +
# Real good_for_kids values aggregated by business
business_photo_good_for_kids_test = pd.DataFrame()
business_photo_good_for_kids_test['business_id'] = business_id_test
business_photo_good_for_kids_test['good_for_kids'] = y_kids_test
business_good_for_kids_test = business_photo_good_for_kids_test.groupby('business_id').agg({'good_for_kids':'mean'})

# Predicted good_for_kids values aggregated by business
business_photo_good_for_kids_test_pred = pd.DataFrame()
business_photo_good_for_kids_test_pred['business_id'] = business_id_test
business_photo_good_for_kids_test_pred['good_for_kids'] = y_kids_test_pred
business_good_for_kids_test_pred = business_photo_good_for_kids_test_pred.groupby('business_id').agg({'good_for_kids':'mean'})
business_good_for_kids_test_pred['good_for_kids'] = business_good_for_kids_test_pred['good_for_kids'] > 0.5

# What percentage of businesses are good for kids?
y_businesses_counts = np.bincount(business_good_for_kids_test.good_for_kids.values)
y_businesses_frequencies = y_businesses_counts / np.sum(y_businesses_counts)
print('class frequencies: {}'.format(y_businesses_frequencies))

# As a baseline compute metrics for a trivial classifier that always outputs the most popular class.
most_frequent_y_businesses = np.full(shape=business_good_for_kids_test.good_for_kids.values.shape,
                                     fill_value=bool(np.argmax(y_businesses_frequencies)))
print('Mean constant accuracy: {}'.format(accuracy_score(business_good_for_kids_test.good_for_kids.values,
                                                         most_frequent_y_businesses)))
print('Balanced mean constant accuracy: {}'.format(balanced_accuracy_score(business_good_for_kids_test.good_for_kids.values,
                                                                           most_frequent_y_businesses)))

# What is the actual metric?

print('Mean test accuracy by business: {}'.format(
    accuracy_score(business_good_for_kids_test.good_for_kids.values,
                   business_good_for_kids_test_pred.good_for_kids.values)))
print('Balanced mean test accuracy by business: {}'.format(
    balanced_accuracy_score(business_good_for_kids_test.good_for_kids.values,
                            business_good_for_kids_test_pred.good_for_kids.values)))
# -

# Balanced accuracy is still better than constant guessing, but unbalnced accuracy is still worse.

# ### Let's finally take is_restaurant as an another example
# We should expect better metrics than for `good_for_kids` as food is directly visible on photos.

# +
# Get is_restaurant feature for all businesses and add it into business_df
is_restaurant = business_df.categories.apply(lambda categories: 'Restaurants' in categories if categories else None)
business_df['is_restaurant'] = is_restaurant

print('is_restaurant stats by business:')
is_restaurant.value_counts(dropna=False)

# +
# Let's merge photos data frame with businesses data frame, 
# but only for businesses where we exactly know if they are restaurants or not.
business_is_restaurant_df = business_df[['business_id', 'is_restaurant']]
business_is_restaurant_df = business_is_restaurant_df[~business_is_restaurant_df.is_restaurant.isnull()]
photo_is_restaurant_df = pd.merge(left=photos_with_features_df_full, right=business_is_restaurant_df, how='inner', on='business_id')
print('is_restaurant stats by photo:')
print(photo_is_restaurant_df.is_restaurant.value_counts(dropna=False))

# Sample 20k of examples and get Xs and ys.
photo_is_restaurant_df = photo_is_restaurant_df.sample(n=20000, random_state=42).reset_index()
photo_is_restaurant_features = np.stack(photo_is_restaurant_df.features.values, axis=0)
photo_is_restaurant_y = photo_is_restaurant_df.is_restaurant.values.astype(np.bool)
#photo_is_restaurant_y_business_id = photo_is_good_for_kids_df.business_id.values
# -

# Even though most businesses are not restaurants, most photos belong to restaurants.

# Split is_restaurant data into train and test datasets for validation.
X_restaurants_train, X_restaurants_test, y_restaurants_train, y_restaurants_test = train_test_split(photo_is_restaurant_features, photo_is_restaurant_y, test_size=0.2, random_state=42)

# +
classifier_restaurants = Pipeline([
    ('scaler', StandardScaler()),
    ('logit', LogisticRegression(solver='lbfgs', C=0.00001, verbose=0, max_iter=2000, 
                                class_weight='balanced', random_state=42))    
    ])

# What percentage of photos belong to restaurants?
y_restaurants_counts = np.bincount(y_restaurants_train)
y_restaurants_frequencies = y_restaurants_counts / np.sum(y_restaurants_counts)
print('class frequencies: {}'.format(y_restaurants_frequencies))

# As a baseline compute metrics for a trivial classifier that always outputs the most popular class.
most_frequent_y_restaurants = np.full(shape=y_restaurants_test.shape, fill_value=bool(np.argmax(y_restaurants_frequencies)))
print('Mean constant accuracy: {}'.format(accuracy_score(y_restaurants_test, most_frequent_y_restaurants)))
print('Balanced mean constant accuracy: {}'.format(balanced_accuracy_score(y_restaurants_test, most_frequent_y_restaurants)))

# and for a classifier that outputs a random class.
np.random.seed(42)
random_y_restaurants = np.random.choice([False, True], size=y_restaurants_test.shape, p=y_restaurants_frequencies)
print('Mean random accuracy: {}'.format(accuracy_score(y_restaurants_test, random_y_restaurants)))
print('Balanced mean random accuracy: {}'.format(balanced_accuracy_score(y_restaurants_test, random_y_restaurants)))

classifier_restaurants = classifier_restaurants.fit(X_restaurants_train, y_restaurants_train)
y_restaurants_train_pred = classifier_restaurants.predict(X_restaurants_train)
print('Mean train accuracy: {}'.format(accuracy_score(y_restaurants_train, y_restaurants_train_pred)))
print('Balanced mean train accuracy: {}'.format(balanced_accuracy_score(y_restaurants_train, y_restaurants_train_pred)))

y_restaurants_test_pred = classifier_restaurants.predict(X_restaurants_test)
y_restaurants_test_pred_proba = classifier_restaurants.predict_proba(X_restaurants_test)[:, 1]
print('Mean test accuracy: {}'.format(accuracy_score(y_restaurants_test, y_restaurants_test_pred)))
print('Balanced mean test accuracy: {}'.format(balanced_accuracy_score(y_restaurants_test, y_restaurants_test_pred)))
print('Test average precision: {}'.format(average_precision_score(y_restaurants_test, y_restaurants_test_pred_proba)))

# -

# As we expected, deep photo features better predict if a place is a restaraunt comparing to if a place is good for kids (both in  terms of balanced accuracy and average precision).


