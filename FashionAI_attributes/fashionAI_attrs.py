#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, time, random, shutil, math
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt

from mxnet import gluon
from mxnet import autograd
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.model_zoo import vision as models


task = "skirt_length_labels"
warmup_label_dir = 'data/web/Annotations/skirt_length_labels.csv'
base_label_dir = 'data/base/Annotations/label.csv'

image_path = []

with open(warmup_label_dir, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, _, label in tokens:
        image_path.append(('data/web/'+path, label))

with open(base_label_dir, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, tk, label in tokens:
        if tk == task:
            image_path.append(('data/base/'+path, label))


def plot_image(img_path):
    with open(img_path, 'rb') as f:
        img = image.imdecode(f.read())
    plt.imshow(img.asnumpy())
    return img


# 分配训练验证集
n = len(image_path)
random.seed(1024)
random.shuffle(image_path)
train_count = 0
for path, label in image_path:
    label_index = list(label).index('y')
    if train_count < n * 0.9:
        shutil.copy(path, os.path.join('data/train_valid', task, 'train', str(label_index)))
    else:
        shutil.copy(path, os.path.join('data/train_valid', task, 'val', str(label_index)))
    train_count += 1


# 迁移学习，定义预训练网络
pretrained_net = models.resnet50_v2(pretrained=True)

num_gpu = 1
ctx = [mx.gpu(i) for i in range(num_gpu)] if num_gpu > 0 else [mx.cpu()]


finetune_net = models.resnet50_v2(classes=6)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier(), ctx=ctx)
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()


# 训练集图像增广
def aug_train(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=256,
                                    rand_crop=True, rand_mirror=True,
                                    mean=np.array([0.485, 0.456, 0.406]),
                                    std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return (im, nd.array([label]).asscalar())


# 验证集图像增广
def aug_val(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=256,
                                    mean=np.array([0.485, 0.456, 0.406]),
                                    std=np.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return (im, nd.array([label]).asscalar())


# 计算average precision
def calculate_ap(labels, outputs):
    cnt = 0
    ap = 0.
    for label, outptu in zip(labels, outputs):
        for lb, op in zip(label.asnumpy().astype(np.int), outputs.asnumpy()):
            op_argsort = np.argsort(op)[::-1]
            lb_int = int(lb)
            ap += 1.0 / (1+list(op_argsort).index(lb_int))
            cnt += 1
    return ((ap, cnt))


# 在训练集上的测试
def validate(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    AP = 0.
    AP_cnt = 0
    val_loss = 0.
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

        outputs = [net(X) for X in data]
        metric.update(label, outputs)
        loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        val_loss = sum([l.mean().asscalar() for l in loss]) / len(loss)
        ap, cnt = calculate_ap(label, outputs)
        AP += ap
        AP_cnt += cnt
    _, val_acc = metric.get()
    return ((val_acc, AP / AP_cnt, val_loss / len(val_data)))


lr = 1e-3
momentum = 0.9
wd = 1e-4
epochs = 5
batch_size = 64


train_path = os.path.join('data/train_valid', task, 'train')
val_path = os.path.join('data/train_valid', task, 'val')

# 训练集的DataLoader
train_data = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(train_path, transform=aug_train),
                                   batch_size=batch_size, shuffle=True, num_workers=4)
# 验证集的DataLoader
val_data = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(val_path, transform=aug_val),
                                 batch_size=batch_size, shuffle=False, num_workers=4)


trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {'learning_rate': lr,
                                           'momentum': momentum, 'wd': wd})
L = gluon.loss.SoftmaxCrossEntropyLoss()
metric = mx.metric.Accuracy()


# train
for epoch in epochs:
    tic = time.time()

    train_loss = 0.
    metric.reset()
    AP = 0.
    AP_cnt = 0

    num_batch = len(train_data)

    for i, batch in enumerate(train_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

        with autograd.record():
            outputs = [finetune_net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()
        trainer.step(batch_size)
        train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

        metric.update(label, outputs)
        ap, cnt = calculate_ap(label, outputs)
        AP += ap
        AP_cnt += cnt

    train_map = AP / AP_cnt
    _, train_acc = metric.get()
    train_loss /= num_batch

    val_acc, val_map, val_loss = validate(finetune_net, val_data, ctx)
    print('[Epoch %d] train-acc: %f, mAP: %f, loss: %f, | val-acc: %f, mAP: %f, loss: %f | time: %f'
          % (epoch, train_acc, train_map, train_loss, val_acc, val_map, val_loss, time.time()()-tic))


