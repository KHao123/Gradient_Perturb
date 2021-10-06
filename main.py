"""
Teacher free KD, main.py
"""
import argparse
import logging
import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils
import model.net as net
import data_loader as data_loader
import model.resnet as resnet
import model.mobilenetv2 as mobilenet
import model.densenet as densenet
import model.resnext as resnext
import model.shufflenetv2 as shufflenet
import model.alexnet as alexnet
import model.googlenet as googlenet
import torchvision.models as models
from my_loss_function import loss_label_smoothing, loss_kd_regularization, loss_kd, loss_kd_self,FocalLoss,loss_kd_t
from train import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/imbalance_experiments/resample_resnet18/', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--num_class', default=100, type=int, help="number of classes")
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')

# parser.add_argument('--focal_loss', action='store_true', default=False, help="flag for focal loss")


def main():
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Set the random seed for reproducible experiments
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    warnings.filterwarnings("ignore")

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    logging.info(params.dataset+" "+str(params.cifar_imb_ratio))

    # fetch dataloaders, considering full-set vs. sub-set scenarios
    
    
    train_dl, cls_num,_ = data_loader.fetch_dataloader('train', params)

    dev_dl,_,_ = data_loader.fetch_dataloader('dev', params)
    args.num_class = cls_num
    logging.info("- done.")

   
    
    print("Train base model")
    if params.model_version == "cnn":
        model = net.Net(params).cuda()

    elif params.model_version == "mobilenet_v2":
        print("model: {}".format(params.model_version))
        model = mobilenet.mobilenetv2(class_num=args.num_class).cuda()

    elif params.model_version == "shufflenet_v2":
        print("model: {}".format(params.model_version))
        model = shufflenet.shufflenetv2(class_num=args.num_class).cuda()

    elif params.model_version == "alexnet":
        print("model: {}".format(params.model_version))
        model = alexnet.alexnet(num_classes=args.num_class).cuda()

    elif params.model_version == "vgg19":
        print("model: {}".format(params.model_version))
        # model = models.vgg19_bn(num_classes=args.num_class).cuda()
        model = models.vgg19_bn(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, args.num_class),
        )
        model = model.cuda()

    elif params.model_version == "googlenet":
        print("model: {}".format(params.model_version))
        model = googlenet.GoogleNet(num_class=args.num_class).cuda()

    elif params.model_version == "densenet121":
        print("model: {}".format(params.model_version))
        model = densenet.densenet121(num_class=args.num_class).cuda()
    elif params.model_version == "densenet169":
        print("model: {}".format(params.model_version))
        model = models.densenet169(True)
        model.classifier = nn.Linear(model.classifier.in_features, args.num_class)
        model = model.cuda()

    elif params.model_version == "resnet18":
        model = resnet.ResNet18(num_classes=args.num_class).cuda()
    elif params.model_version == "resnet32":
        model = resnet.resnet32(num_classes=args.num_class).cuda()

    elif params.model_version == "resnet50":
        # model = resnet.ResNet50(num_classes=args.num_class).cuda()
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.num_class)
        model = model.cuda()

    elif params.model_version == "resnet101":
        model = resnet.ResNet101(num_classes=args.num_class).cuda()

    elif params.model_version == "resnet152":
        model = resnet.ResNet152(num_classes=args.num_class).cuda()

    elif params.model_version == "resnext29":
        model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=args.num_class).cuda()
        # model = nn.DataParallel(model).cuda()

    
    loss_fn = nn.CrossEntropyLoss()
        
    
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9,
                            weight_decay=5e-4)

    iter_per_epoch = len(train_dl)
    warmup_scheduler = utils.WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, params,
                        args.model_dir, warmup_scheduler, args, args.restore_file)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES']='0'
    main()

