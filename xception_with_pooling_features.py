import torch.nn.functional as F
import torch.nn as nn

from pretrainedmodels.models.xception import Xception, pretrained_settings
from torch.utils import model_zoo


class XceptionWithPoolingFeatures(Xception):
    """
    Passes features through ReLU and average pooling reducing their count from 2048*10*10 to just 2048.
    """

    def features(self, input):
        base_features = super().features(input)
        features = nn.ReLU(inplace=True)(base_features)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        return features

    def logits(self, features):
        x = super().logits(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

def xception_with_pooling_features(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = XceptionWithPoolingFeatures(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model
