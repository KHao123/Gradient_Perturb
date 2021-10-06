# import large_models
# import small_models
import baseline
# import vib_models
import torch
import torch.nn as nn
from torchvision import models


def get_model(type='resnet18', no_mahala=True, classes=7):
    if not no_mahala:
        # if type == 'resnet50':
        #     model = large_models.resnet50(num_classes=classes)
        # if type == 'resnet18':
        #     model = large_models.resnet18(num_classes=classes)
        # if type == 'resnet20':
        #     model = small_models.resnet20(num_classes=classes, trans=True)
        # if type == 'resnet32':
        #     model = small_models.resnet32(num_classes=classes, trans=True)
        # if type == 'resnet56':
        #     model = small_models.resnet56(num_classes=classes, trans=True)
        # # model.fc = nn.Linear(model.fc.in_features, classes)
        pass
    else:
        if type == 'resnet50':
            model = models.resnet50(False)
            model.load_state_dict(torch.load('../.cache/torch/checkpoints/resnet50-19c8e357.pth'))
        if type == 'resnet18':
            model = models.resnet18(False)
            model.load_state_dict(torch.load('../.cache/torch/checkpoints/resnet18-5c106cde.pth'))
        if type == 'resnet20':
            model = baseline.resnet20()
        if type == 'resnet32':
            model = baseline.resnet32()
        if type == 'resnet56':
            model = baseline.resnet56()
        model.fc = nn.Linear(model.fc.in_features, classes)
    return model

# def get_vib_model(type='resnet18'):
#     if type == 'resnet32':
#         model = vib_models.resnet32(num_classes=10)
#     if type == 'resnet20':
#         model = vib_models.resnet20(num_classes=10)
#     return model