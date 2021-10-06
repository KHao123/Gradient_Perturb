"""Evaluates the model"""

import argparse
import logging

from torch.autograd import Variable
import utils
import data_loader
import os
from model import resnet
import torch
from sklearn import metrics
import utils
import logging
import random
import numpy as np
from torchvision import models
import torch.nn as nn
from calData import testData




def evaluate(model, loss_fn, dataloader, params, args):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()
    losses = utils.AverageMeter()
    total = 0
    correct = 0
    y_true, y_pred = [], []
    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()

        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        # compute model output
        output_batch = model(data_batch)

        _, predicted = output_batch.max(1)
        # print(data_batch.size(0))
        y_true.append(labels_batch.cpu())
        y_pred.append(predicted.cpu())
        

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    logging.info('\n'+metrics.classification_report(y_true, y_pred, digits=3))

    # print(metrics.recall_score(y_true, y_pred,average='macro'))
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    f1 = metrics.f1_score(y_true, y_pred,average='macro')
    return recall, precision, f1

def evaluate_perturb(model,loss_fn,dataloader,img_num,epsilon=0.0):
    
    model.eval()
    # logging.info(img_num)
    img_num = img_num.cuda()
    # weights = torch.flip(weights, dims=(0,))
    weights = 1.0 / img_num
    # weights = img_num
    weights = torch.log(weights)
    weights = weights - min(weights)
    weights /= weights.sum()
    logging.info(weights)

    y_true, y_pred = [], []

    T = 1000
    
    for (x, label) in dataloader:
        x = x.cuda()
        x.requires_grad_()
        logits = model(x)

        loss = loss_fn(logits / T, weights)
        loss.backward()

        with torch.no_grad():
            
            x -= epsilon * torch.sign(x.grad)
        
        logits = model(x)
        pred = logits.argmax(dim=1)
        y_true.append(label)
        y_pred.append(pred.cpu())
    # print('*'*20)
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    
    logging.info('\n'+metrics.classification_report(y_true, y_pred, digits=3))

    # print(metrics.recall_score(y_true, y_pred,average='macro'))
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    f1 = metrics.f1_score(y_true, y_pred,average='macro')
    return recall, precision, f1

def cross_entropy(logits, label):
    logits = torch.log_softmax(logits, dim=1)
    label = label.unsqueeze(0).repeat(logits.size(0), 1)
    
    # label = torch.mul(label,epsilons)
    loss = -torch.sum(logits * label) / logits.size(0)
    return loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='experiments/imbalance_experiments/resample_resnet18/', help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir \
                        containing weights to reload before training")  # 'best' or 'train'
    parser.add_argument('--num_class', default=100, type=int, help="number of classes")
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--epsilon', type=float, default=0.00, help='warm up training phase')
    parser.add_argument('--regularization', action='store_true', default=False, help="flag for regulization")
    parser.add_argument('--label_smoothing', action='store_true', default=False, help="flag for label smoothing")
    parser.add_argument('--double_training', action='store_true', default=False, help="flag for double training")
    parser.add_argument('--self_training', action='store_true', default=False, help="flag for self training")
    parser.add_argument('--pt_teacher', action='store_true', default=False, help="flag for Defective KD")
    parser.add_argument('--tem', action='store_true', default=False, help="flag for self training")

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    recalls = []
    precisions =[]
    f1s = []
    epsilon = args.epsilon
    temper = 1000
    criterion = nn.CrossEntropyLoss()
    
    model_dir = ""

    cross_val = False
    if 'skin7' in args.model_dir or 'sd198' in args.model_dir:
        cross_val = True 

    if cross_val:
        if epsilon == 0:
            utils.set_logger(os.path.join(args.model_dir+'_iterNo1', 'normal_test.log'))
        else:
            utils.set_logger(os.path.join(args.model_dir+'_iterNo1', 'perturb_test.log'))
    else:
        if epsilon == 0:
            utils.set_logger(os.path.join(args.model_dir, 'normal_test.log'))
        else:
            utils.set_logger(os.path.join(args.model_dir, 'perturb_test.log'))

    logging.info("\n#####################test epsilon={}##################\n".format(epsilon))

    if cross_val:
        iters = 5
    else:
        iters = 1

    for i in range(iters):
        if cross_val:
            iterNo = "_iterNo" + str(i+1)
        else:
            iterNo = ""
        model_dir = args.model_dir + iterNo
        json_path = os.path.join(model_dir, 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = utils.Params(json_path)
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed(0)
        
        train_loader,cls_num,img_num = data_loader.fetch_dataloader('train', params)
        dev_dl,_,_ = data_loader.fetch_dataloader('dev', params)
        args.num_class = cls_num
        if params.model_version == "cnn":
            model = net.Net(params).cuda()

        elif params.model_version == "mobilenet_v2":
            print("model: {}".format(params.model_version))
            model = models.mobilenet_v2(pretrained=True)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.last_channel, args.num_class),
            )
            model = model.cuda()

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


        checkpoint = torch.load(os.path.join(model_dir,'best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        
        
       
        recall, precision, f1 = evaluate_perturb(model,cross_entropy,dev_dl,img_num,epsilon)


        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    recalls = [x * 100 for x in recalls]
    precisions = [x * 100 for x in precisions]
    f1s = [x * 100 for x in f1s]

    logging.info(recalls)
    logging.info(precisions)
    logging.info(f1s)
    logging.info("\n mean_recall: {:.2f} var:{:.2f}\n mean_precision: {:.2f} var:{:.2f}\n mean_f1: {:.2f} var:{:.2f}\n".format(np.mean(recalls), np.var(recalls), np.mean(precisions), np.var(precisions), np.mean(f1s), np.var(f1s)))
