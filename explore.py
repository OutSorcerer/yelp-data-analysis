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

# # What information we can get from photos of businesses?
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

from xception_with_pooling_features import xception_with_pooling_features
xception_model = xception_with_pooling_features(num_classes=1000, pretrained='imagenet')

# +
import torch
import pretrainedmodels.utils as utils

load_img = utils.LoadImage()

# transformations depending on the model
# rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
tf_img = utils.TransformImage(xception_model) 

#path_img = 'photos/photos/3-T7FTL_sB4mH2bc3WnPUg.jpg'
path_img = 'photos/photos/3T449kzPrsipjBx7I8ENag.jpg'

input_img = load_img(path_img)
input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor,
    requires_grad=False)

#output_logits = model(input) # 1x1000

output_features = xception_model.features(input) # 1x14x14x2048 size may differ
# -

output_features.shape

output_features



