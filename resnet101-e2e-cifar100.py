import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata
from torch.autograd import Variable
from dataloaders8 import dataloaders
from utils6 import *
from torchsummary import summary
import time, math, numpy as np, matplotlib.pyplot as plt, argparse, os, collections, sys, inspect, pprint, scipy.stats as st
from functools import partial


__all__ = ['ResNet50', 'ResNet101','ResNet152']

def Conv1(in_planes, places):
  return nn.Sequential(
    nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size = 3, stride = 1, padding = 1, bias=False),
    nn.BatchNorm2d(places),
    nn.ReLU(inplace=True),
    #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
  )

class Bottleneck(nn.Module):
  def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
    super(Bottleneck,self).__init__()
    self.expansion = expansion
    self.downsampling = downsampling

    self.bottleneck = nn.Sequential(
      nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
      nn.BatchNorm2d(places),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
      nn.BatchNorm2d(places),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
      nn.BatchNorm2d(places*self.expansion),
    )

    if self.downsampling:
      self.downsample = nn.Sequential(
        nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(places*self.expansion)
      )
    self.relu = nn.ReLU(inplace=True)
  def forward(self, x):
    residual = x
    out = self.bottleneck(x)

    if self.downsampling:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self,blocks, num_classes=1000, expansion = 4, gain = 0.1):
    super(ResNet,self).__init__()
    self.expansion = expansion

    self.conv1 = Conv1(in_planes = 3, places= 64)

    self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
    self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
    self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
    self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

    self.avgpool = nn.AvgPool2d(4)
    self.fc = nn.Linear(2048,num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain = gain)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def make_layer(self, in_places, places, block, stride):
    layers = []
    layers.append(Bottleneck(in_places, places,stride, downsampling =True))
    for i in range(1, block):
      layers.append(Bottleneck(places*self.expansion, places))

    return nn.Sequential(*layers)


  def forward(self, x):
    x = self.conv1(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

def ResNet50():
  return ResNet([3, 4, 6, 3])

def ResNet101():
  return ResNet([3, 4, 23, 3], 100)

def ResNet152():
  return ResNet([3, 8, 36, 3], 500)

def train_e2e(model, optimizer, scheduler, criterion, nepochs, trainloader, valloader, testloader):
  t0, train_loss, val_accuracy = time.time(), [], []
  print('\n--- Begin e2e trainning\n')
  for e in range(nepochs):
    model.train()
    t1, loss_meter, accuracy_meter = time.time(), AverageMeter(), AverageMeter()
    for j, (x, y) in enumerate(trainloader):
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      out = model(x)
      loss = criterion(out, y) 
      loss.backward()
      optimizer.step()
      _, pred = torch.max(out.data, 1)
      update_meters(y, pred, loss.item(), loss_meter, accuracy_meter)
    if scheduler is not None:
      scheduler.step()
    epoch_val_acc = test_e2e(model, criterion, testloader)
    train_loss.append(loss_meter.avg)
    val_accuracy.append(epoch_val_acc)
    m = (e + 1, nepochs, loss_meter.avg, epoch_val_acc, time.time() - t1, time.time() - t0)
    print('\n[***** Ep {:^5}/{:^5} over ******] Train loss {:.4f} Valid acc {:.4f} Epoch time {:9.4f}s Total time {:.4f}s\n'.format(*m))
  return train_loss, val_accuracy


def test_e2e(model, criterion, loader):
  model.eval()
  loss_meter, accuracy_meter = AverageMeter(), AverageMeter()
  for j, (x, y) in enumerate(loader):
    with torch.no_grad():
      x, y = x.to(device), y.to(device)
      out = model(x)
      loss = criterion(out, y)
      _, pred = torch.max(out.data, 1)
      update_meters(y, pred, loss.item(), loss_meter, accuracy_meter)
  return accuracy_meter.avg

if __name__=='__main__':
  seed = 2
  if seed is not None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = ResNet101()
  #print(model)
  #summary(model, (3, 64, 64))

  dataset = 'cifar100'
  batchsize = 256
  trainloader, valloader, testloader, datashape, nclasses, datamean, datastd = dataloaders(dataset, batchsize, None, None, None)
  opt = 'sgd'
  lrt = 0.1
  lrd = 1
  m = 0.9
  w = 0.0001
  lbs = 0.1
  nepochs = 300
  print(seed, opt, lrt, lbs, nepochs)
  criterion = nn.CrossEntropyLoss(label_smoothing = lbs)
  optimizer = optim.SGD(model.parameters(), lr = lrt, momentum = m, weight_decay = w) if opt == 'sgd' else optim.Adam(model.parameters(), lr = lrt, betas = (be1, be2))
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [120, 160, 200], gamma = 0.2) if opt == 'sgd' and lrd else None
  model.to(device)
  train_loss, val_accuracy = train_e2e(model, optimizer, scheduler, criterion, nepochs, trainloader, valloader, testloader)
  print('max val acc', max(val_accuracy))
  