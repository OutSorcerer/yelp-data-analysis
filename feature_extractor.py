"""
Extract features from all photos and caches the result in a file.
"""
import os

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from xception_with_pooling_features import xception_with_pooling_features
import pretrainedmodels.utils as utils

def extract():
    cuda_is_availabe = torch.cuda.is_available()
    print('CUDA is available' if cuda_is_availabe else 'CUDA is NOT available')
    xception_model = xception_with_pooling_features(num_classes=1000, pretrained='imagenet')
    if cuda_is_availabe:
        xception_model = xception_model.to(torch.device('cuda:0'))
    photo_df = pd.read_json('photos/photo.json', lines=True)

    load_img = utils.LoadImage()

    # transformations depending on the model
    # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
    tf_img = utils.TransformImage(xception_model)

    photo_df['features'] = None
    # Extract features just for a small subset.
    #photo_df = photo_df.loc[:100-1,:]
    for index, row in tqdm(photo_df.iterrows(), total=photo_df.shape[0], desc="Extracting features from photos"):
        photo_id = row['photo_id']
        file_name = os.path.join('photos', 'photos', photo_id + '.jpg')
        input_img = load_img(file_name)
        input_tensor = tf_img(input_img)  # 3x?x? -> 3x299x299 size may differ
        input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
        if cuda_is_availabe:
            input_tensor = input_tensor.cuda()
        with torch.no_grad():
            #input = torch.autograd.Variable(input_tensor, requires_grad=False)
            output_features = xception_model.features(input_tensor).cpu().numpy()  # 1x2048x1x1
        output_features = np.reshape(output_features, (-1,))
        row['features'] = output_features

    os.makedirs('.cache', exist_ok=True)
    photo_df.to_pickle('.cache/photo.pkl')


def load_photos_with_features_df():
    if not os.path.isfile('.cache/photo.pkl'):
        extract()
    return pd.read_pickle('.cache/photo.pkl')


if __name__== "__main__":
  load_photos_with_features_df()
