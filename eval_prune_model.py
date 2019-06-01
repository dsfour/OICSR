from __future__ import print_function
import os
import torch
import numpy as np
from os.path import join as join
from os.path import exists as exist
from os import mkdir as mkdir
import copy
import time
import argparse
import collections
import torch.nn as nn
from torchvision import datasets, transforms
import tensorflow as tf
import models
import random
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--test_data', type=str, default='/mnt/cephfs_wj/cv/common/datasets/ImageNet/ILSVRC2012_img_val',
                    help='path to ImageNet dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--fpp', type=float, default=0,
                    help='Flops pruned percent of model')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def validate(val_loader, model, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                input = input.cuda()
                # compute output
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))

            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        return top1.avg
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def validate(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input = input.cuda()
            # compute output
            output = model(input)
            loss = nn.CrossEntropyLoss()(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg
def cal_model_flops(model):
     model.eval()
     input = torch.ones([1, 3, 224, 224], dtype = torch.float32).cuda()
     flops_list=[]
     def conv_hook(self, input, output):
         output_channels, output_height, output_width = output[0].size()
         flops = (self.out_channels/self.groups) * (self.kernel_size[0] * self.kernel_size[1] *self.in_channels/self.groups) * output_height * output_width*self.groups
         flops_list.append(flops)

     def linear_hook(self, input, output):
         flops = self.in_features*self.out_features
         flops_list.append(flops)

     def foo(net):
         childrens = list(net.children())
         if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            return
         for c in childrens:
            foo(c)
     foo(model)
     output = model(input)
     flops = sum(flops_list)
     return flops


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.test_data, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.test_batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

def main():

    if args.fpp==0:
        model_path = "checkpoints/resnet50.pth.tar"
        checkpoint = torch.load(model_path)
        cfg = [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256,
               256, 256,
               256, 256, 512, 512, 512, 512, 512, 512]
    else:
        model_path = "checkpoints/resnet50-%.1f%%FLOPs.pth.tar"%args.fpp
        checkpoint = torch.load(model_path)
        cfg = checkpoint['cfg']
    model = models.__dict__[args.arch](cfg=cfg)
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])

    total_flops= cal_model_flops(model)
    print("model %s, total_flops: %d"%(model_path, total_flops))
    validate(test_loader, model)


if __name__ == '__main__':
    main()
