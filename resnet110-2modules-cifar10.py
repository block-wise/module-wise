# even splits
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as functional, torch.utils.data as torchdata
from torch.autograd import Variable
from dataloaders8 import dataloaders
from utils7 import *
from torchsummary import summary
import time, math, numpy as np, matplotlib.pyplot as plt, argparse, os, collections, sys, inspect, pprint, scipy.stats as st
from functools import partial

class Downsample(nn.Module):  
	def __init__(self):
		super(Downsample, self).__init__() 
		self.avg = nn.AvgPool2d(1, 2)   
	def forward(self, x):   
		x = self.avg(x)  
		return torch.cat((x, x.mul(0)), 1)  

class ResBlock(nn.Module):
	def __init__(self, first, infilters, nfilters, stride = 1, downsampling = False):
		super(ResBlock, self).__init__()
		self.first, self.downsampling = first, downsampling
		if self.first:
			self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias = False), nn.BatchNorm2d(16), nn.ReLU(inplace = True))
		self.residual = nn.Sequential(nn.Conv2d(infilters, nfilters, 3, stride, 1, bias = False), nn.BatchNorm2d(nfilters), nn.ReLU(inplace = True),
      								  nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = False), nn.BatchNorm2d(nfilters))
		if self.downsampling:
			self.downsample = Downsample()
		self.relu = nn.ReLU(inplace = True)
	def forward(self, x):
		if self.first:
			x = self.encoder(x)
		z = self.residual(x)
		if self.downsampling:
			x = self.downsample(x)	
		return self.relu(x + z), z

class ResModule1(nn.Module):
	def __init__(self, featureshape, nclasses, clname, apc, initialization):
		super(ResModule1, self).__init__()
		blocks1 = [ResBlock(True, 16, 16, 1, False) if i == 0 else ResBlock(False, 16, 16) for i in range(18)]
		blocks2 = [ResBlock(False, 16, 32, 2, True) if i == 0 else ResBlock(False, 32, 32) for i in range(9)]
		self.blocks = nn.ModuleList(blocks1 + blocks2)
		self.blocks.apply(initialization)
		self.classifier = create_classifier(clname, nclasses, featureshape, apc)
		self.classifier.apply(initialization)
	def forward_conv(self, x):
		rs = []
		for block in self.blocks:
			x, r = block(x)
			rs.append(r)
		return x, rs
	def forward(self, x):
		x, rs = self.forward_conv(x)
		out = self.classifier(x)
		return out, x, rs

class ResModule2(nn.Module):
	def __init__(self, featureshape, nclasses, clname, apc, initialization):
		super(ResModule2, self).__init__()
		blocks1 = [ResBlock(False, 32, 32) for i in range(9)]
		blocks2 = [ResBlock(False, 32, 64, 2, True) if i == 0 else ResBlock(False, 64, 64) for i in range(18)]
		self.blocks = nn.ModuleList(blocks1 + blocks2)
		self.blocks.apply(initialization)
		self.classifier = create_classifier(clname, nclasses, featureshape, apc)
		self.classifier.apply(initialization)
	def forward_conv(self, x):
		rs = []
		for block in self.blocks:
			x, r = block(x)
			rs.append(r)
		return x, rs
	def forward(self, x):
		x, rs = self.forward_conv(x)
		out = self.classifier(x)
		return out, x, rs



def train_submodel(totrain, modules, optimizers, schedulers, criterion, tra, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, 
				   trainloader, valloader, testloader, r = None):
	print('\n' + '-' * 64, f'Round {r}' if r is not None else '', 'Submodel', totrain)
	train_loss, train_accuracy, val_accuracy, it = [], [], [], 0
	if lml0type == 'decreasing':
		lml, lmt = lml0 / totrain ** lml0power if totrain > 0 else lml0, totrain ** lml0power / lml0
	elif lml0type == 'increasing':
		lml, lmt =  lml0 * totrain ** lml0power, 1 / (lml0 * totrain ** lml0power) if totrain > 0 else 1 / lml0
	for epoch in range(1, ne1 + totrain * ne2 + 1):
		for module in modules:
			module.train()
		t1, loss_meter, accuracy_meter = time.time(), AverageMeter(), AverageMeter()
		for j, (x, y) in enumerate(trainloader):
			it = it + 1
			x, y = x.to(device), y.to(device)
			z = Variable(x.data, requires_grad = False).detach()
			for i in range(totrain + 1):
				optimizers[i].zero_grad()
				out, w, rs = modules[i](z)
				z = Variable(w.data, requires_grad = False).detach()
				if i == totrain:
					target = criterion(out, y)
					if tra or uza :
						transport = sum([torch.mean(r ** 2) for r in rs]) 
					if uza and it % uzs == 0 :
						lml += uzt * target.item()
						lmt = 1 / lml
					loss = target + transport / (2 * taus[i]) if tra else (target + lmt * transport if uza else target)
					loss.backward()
					optimizers[i].step()
					if schedulers is not None:
						schedulers[i].step()
					_, pred = torch.max(out.data, 1)
					update_meters(y, pred, target.item(), loss_meter, accuracy_meter)
		epoch_train_loss, epoch_train_accuracy, epoch_val_accuracy = loss_meter.avg, accuracy_meter.avg, test_submodel(totrain, models, criterion, testloader)
		print('\n' + '-' * 64, f'Round {r}' if r is not None else '', 'Submodel', totrain, 'Epoch', epoch, 'Took', time.time() - t1, 's')
		print('Transport', tra, 'tau =', taus[totrain], 'Uzawa', uza, 'lmt =', lmt)
		print('Train loss', epoch_train_loss, 'Train accuracy', epoch_train_accuracy, 'Val accuracy', epoch_val_accuracy)
		train_loss.append(epoch_train_loss)
		train_accuracy.append(epoch_train_accuracy)
		val_accuracy.append(epoch_val_accuracy)
	return train_loss, val_accuracy

def train_seq(modules, optimizers, schedulers, criterion, tra, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, 
			  trainloader, valloader, testloader, r = None):
	train_loss, train_accuracy, val_accuracy = [], [], []
	for totrain in range(len(modules)):
		trloss, vlacc = train_submodel(totrain, modules, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, trainloader, valloader, testloader, r)
	return trloss, vlacc

def test_submodel(totest, modules, criterion, loader):
	loss_meter, accuracy_meter = AverageMeter(), AverageMeter()
	for module in modules:
		module.eval()
	for j, (x, y) in enumerate(loader):
		x, y = x.to(device), y.to(device)
		z = Variable(x.data, requires_grad = False).detach()
		for i in range(totest + 1) :
			with torch.no_grad():
				out, w, rs = modules[i](z)
				z = Variable(w.data, requires_grad = False).detach()
				if i == totest:
					target = criterion(out, y)
					_, pred = torch.max(out.data, 1)
					update_meters(y, pred, target.item(), loss_meter, accuracy_meter)
	return accuracy_meter.avg

def train_par(modules, optimizers, schedulers, criterion, tra, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne0, trainloader, valloader, testloader):
	t0, nmodules, train_loss, train_accuracy, val_accuracy, it = time.time(), len(modules), [], [], [], 0 
	if lml0type == 'decreasing':
		lmls, lmts = [lml0 / i ** lml0power if i > 0 else lml0 for i in range(nmodules)], [i ** lml0power / lml0 for i in range(nmodules)] 
	elif lml0type == 'increasing':
		lmls, lmts =  [lml0 * i ** lml0power for i in range(nmodules)], [1 / (lml0 * i ** lml0power) if i > 0 else 1 / lml0 for i in range(nmodules)] 
	print('parallel training for', ne0, 'epochs')
	for epoch in range(1, ne0 + 1):
		for module in modules:
			module.train()
		t1, loss_meters, accuracy_meters = time.time(), [AverageMeter() for _ in range(nmodules)], [AverageMeter() for _ in range(nmodules)]
		for j, (x, y) in enumerate(trainloader):
			it = it + 1
			x, y = x.to(device), y.to(device)
			z = Variable(x.data, requires_grad = False).detach()
			for i, module in enumerate(modules):
				optimizers[i].zero_grad()
				out, w, rs = module(z)
				z = Variable(w.data, requires_grad = False).detach()
				target = criterion(out, y)
				if tra or uza :
					transport = sum([torch.mean(r ** 2) for r in rs]) 
				if uza and it % uzs == 0 :
					lmls[i] += uzt * target.item()
					lmts[i] = 1 / lmls[i]
				loss = target + transport / (2 * taus[i]) if tra else (target + lmts[i] * transport if uza else target) 
				loss.backward()
				optimizers[i].step()
				if schedulers is not None:
					schedulers[i].step()
				_, pred = torch.max(out.data, 1)
				update_meters(y, pred, target.item(), loss_meters[i], accuracy_meters[i])
		epoch_val_accuracies = test_par(modules, criterion, testloader)
		epoch_train_losses, epoch_train_accuracies = [loss_meters[i].avg for i in range(nmodules)], [accuracy_meters[i].avg for i in range(nmodules)]
		print('-' * 64, 'Epoch', epoch, 'took', time.time() - t1, 's')
		print('Transport', tra, 'taus', taus, 'Uzawa', uza, 'lmts', lmts)
		print('Train losses', epoch_train_losses, '\nTrain accuracies', epoch_train_accuracies, '\nVal accuracies', epoch_val_accuracies)
		train_loss.append(np.max(epoch_train_losses))
		train_accuracy.append(np.max(epoch_train_accuracies))
		val_accuracy.append(np.max(epoch_val_accuracies))
	return train_loss, val_accuracy

def test_par(modules, criterion, loader):
	nmodules = len(modules)
	loss_meters, accuracy_meters = [AverageMeter() for _ in range(nmodules)], [AverageMeter() for _ in range(nmodules)]
	for module in modules:
		module.eval()
	for j, (x, y) in enumerate(loader):
		x, y = x.to(device), y.to(device)
		z = Variable(x.data, requires_grad = False).detach()
		for i, module in enumerate(modules):
			with torch.no_grad():
				out, w, rs = module(z)
				z = Variable(w.data, requires_grad = False).detach()
				target = criterion(out, y)
				_, pred = torch.max(out.data, 1)
				update_meters(y, pred, target.item(), loss_meters[i], accuracy_meters[i])
	return [accuracy_meters[i].avg for i in range(nmodules)]

def train_mro(modules, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, nrounds, 
			  trainloader, valloader, testloader):
	for r in range(1, nrounds + 1):
		print('\n' + '-' * 64, 'Round', r)
		trloss, vlacc = train_seq(modules, nblocks, optimizers, schedulers, criterion, tra, mean, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, 
								  trainloader, valloader, testloader, r)
	return trloss, vlacc

def train_modulewise(traintype, modules, optimizers, schedulers, criterion, tra, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne0, ne1, ne2, nrounds, 
				    trainloader, valloader, testloader):
	if traintype == 'seq':
		return train_seq(modules, optimizers, schedulers, criterion, tra, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, 
						 trainloader, valloader, testloader)
	if traintype == 'par':
		return train_par(modules, optimizers, schedulers, criterion, tra, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne0, 
			   		     trainloader, valloader, testloader)
	elif traintype == 'mro':
		return train_mro(modules, optimizers, schedulers, criterion, tra, taus, uza, lml0, lml0type, lml0power, uzt, uzs, ne1, ne2, nrounds, 
						 trainloader, valloader, testloader)


def modulewise_exp(traintype, featureshape, nclasses, clname, apc, initialization, optimizer, labelsmoothing, learningrate, learningratedecay, beta1, beta2, 
				   transport, tau, varyingtau, lambdaloss0, lambdaloss0type, lambdaloss0power, uzawatau, uzawasteps, uzawa, nepochs0, nepochs1, nepochs2, nrounds, trainloader, valloader, testloader):
	ResModule = {0 : ResModule1, 1 : ResModule2}
	modules = [ResModule[i](featureshape(i), nclasses, clname, apc[i], initialization) for i in range(2)]
	if optimizer == 'adam':
		optimizers = [optim.Adam(filter(lambda p : p.requires_grad, module.parameters()), lr = learningrate, betas = (beta1, beta2)) for module in modules] 
		schedulers = None
	elif optimizer == 'sgd':
		optimizers = [optim.SGD(filter(lambda p : p.requires_grad, module.parameters()), lr = learningrate, momentum = 0.9, weight_decay = 0.0002) for module in modules] 
		schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, 30000, 0.001) for optimizer in optimizers] if learningratedecay else None
	nmodules = len(modules)
	taus = [tau / 2] * int(nmodules / 2) + [tau] * int(nmodules / 2) if varyingtau else [tau] * nmodules
	criterion = nn.CrossEntropyLoss(label_smoothing = labelsmoothing)
	for module in modules:
		module.to(device)
	train_loss, val_accuracy = train_modulewise(traintype, modules, optimizers, schedulers, criterion, transport, taus, uzawa, lambdaloss0, lambdaloss0type, 
												lambdaloss0power, uzawatau, uzawasteps, nepochs0, nepochs1, nepochs2, nrounds, trainloader, valloader, testloader)
	for module in modules:
		del module
	return train_loss, val_accuracy
	

def experiment(batchsize, traintype, clname, initname, initgain, optimizer, labelsmoothing, learningrate, learningratedecay, beta1, beta2, transport,
			   tau, varyingtau, lambdaloss0, lambdaloss0type, lambdaloss0power, uzawatau, uzawasteps, nepochs0, nepochs1, nepochs2, nrounds, trainsize, valsize, 
			   testsize, seed, experiments):

	t0 = time.time()
	dataset = 'cifar10'
	if seed is not None:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.manual_seed(seed)
		np.random.seed(seed)
	uzawa = 1 if (uzawatau > 0 and uzawasteps > 0) else 0
	nepochs = nepochs0 if traintype in ['par', 'e2e'] else nepochs1
	if varyingtau and tau > 0 and not transport:
		transport = 1
		print('transport set to True because varyingtau')
	if not transport and tau > 0:
		tau = 0
		print('no transport despite tau > 0 because transport is False')
	if transport and uzawa:
		transport = 0
		print('transport set to False because uzawa')
	if experiments and nepochs > 1:
		expname = [f'trs{trainsize}', f'tra{transport}', f'uza{uzawa}', f'vta{varyingtau}', f'tau{tau}']
		if uzawa:
			expname = expname + [f'lml0{lambdaloss0}', lambdaloss0type, f'lml0p{lambdaloss0power}', f'uzt{uzawatau}', f'uzs{uzawasteps}']
		stdout0 = sys.stdout
		d = {'par': str(nepochs0), 'e2e': str(nepochs0), 'seq': str(nepochs1) + '-' + str(nepochs2), 'mro': str(nrounds) + '-' + str(nepochs1) + '-' + str(nepochs2)}
		expname = ['log', 'res110-2mod', dataset, traintype, d[traintype], clname, initname, f'ing{initgain}', optimizer, f'lrt{learningrate}', 
				   f'lrd{learningratedecay}'] + expname
		sys.stdout = open('-'.join(expname + [time.strftime("%Y%m%d-%H%M%S")]) + '.txt', 'wt')

	frame = inspect.currentframe()
	names, _, _, values = inspect.getargvalues(frame)
	print('experiment from res110-2mod.py with parameters')
	for name in names:
		print('%s = %s' % (name, values[name]))

	trainloader, valloader, testloader, datashape, nclasses, datamean, datastd = dataloaders(dataset, batchsize, trainsize, valsize, testsize)
	initialization = partial(initialize, initname, initgain)
	encoder = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias = False), nn.BatchNorm2d(16), nn.ReLU(inplace = True))
	encodingshape =  list(encoder(torch.ones(*datashape)).shape)
	nfilters = [32, 64]
	factor = [2, 4]
	featureshape = lambda i : [1, nfilters[i], int(encodingshape[2] / factor[i]), int(encodingshape[3] / factor[i])]
	apc = [8, 8]
	print('train batches', len(trainloader), 'val batches', len(valloader), 'batchsize', batchsize, 'encodingshape', encodingshape, 'apc', apc)


	trloss, vlacc =  modulewise_exp(traintype, featureshape, nclasses, clname, apc, initialization, optimizer, labelsmoothing, learningrate, learningratedecay, beta1, beta2, transport, tau, varyingtau, 
									lambdaloss0, lambdaloss0type, lambdaloss0power, uzawatau, uzawasteps, uzawa, nepochs0, nepochs1, nepochs2, nrounds, trainloader, valloader, testloader)

	if experiments and nepochs > 1:
		print('--- train loss \n', trloss, '\n--- val acc \n', vlacc)
		print('--- min train loss \n', min(trloss), '\n--- max val acc \n', max(vlacc))
		sys.stdout.close()
		sys.stdout = stdout0
	return trloss, vlacc, time.time() - t0

def experiments(parameters, average):
	t0, j, f, accs, nparameters = time.time(), 0, 110, [], len(parameters) 
	nexperiments = int(np.prod([len(parameters[i][1]) for i in range(nparameters)]))
	sep = '-' * f 
	print('\n' + sep, 'res110-2mod.py')
	print(sep, nexperiments, 'res110-2mod experiments ' + ('to average ' if average else '') + 'over parameters:')
	pprint.pprint(parameters, width = f, compact = True)
	for params in product([values for name, values in parameters]) :
		j += 1
		print('\n' + sep, 'res110-2mod experiment %d/%d with parameters:' % (j, nexperiments))
		pprint.pprint([parameters[i][0] + ' = ' + str(params[i]) for i in range(nparameters)], width = f, compact = True)
		train_loss, val_accuracy, t1 = experiment(*params, True)
		accs.append(np.max(val_accuracy))
		print(sep, 'res110-2mod experiment %d/%d over. took %.1f s. total %.1f s' % (j, nexperiments, t1, time.time() - t0))
	if average:
		acc = np.mean(accs)
		confint = st.t.interval(0.95, len(accs) - 1, loc = acc, scale = st.sem(accs))
		print('\nall val acc', accs)
		print('\naverage val acc', acc)
		print('\nconfint', confint)
	print(('\n' if not average else '') + sep, 'total time for %d experiments: %.1f s' % (j, time.time() - t0))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-bas", "--batchsize", type = int, default = [128], nargs = '*')
	parser.add_argument("-trt", "--traintype", default = ['seq'], choices = ['seq', 'par', 'mro', 'e2e'], nargs = '*')
	parser.add_argument("-cln", "--clname", default = ['1CNN'], choices = ['1LIN', '2LIN', '3LIN', '1CNN', 'MPCL', 'MLPS'], nargs = '*')
	parser.add_argument("-inn", "--initname", default = ['orthogonal'], choices = ['orthogonal', 'normal', 'kaiming'], nargs = '*')
	parser.add_argument("-ing", "--initgain", type = float, default = [0.05], nargs = '*')
	parser.add_argument("-opt", "--optimizer", default = ['sgd'], choices = ['adam', 'sgd'], nargs = '*')
	parser.add_argument("-lbs", "--labelsmoothing", type = float, default = [0], nargs = '*')
	parser.add_argument("-lrt", "--learningrate", type = float, default = [0.007], nargs = '*')
	parser.add_argument("-lrd", "--learningratedecay", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-be1", "--beta1", type = float, default = [0.9], nargs = '*')
	parser.add_argument("-be2", "--beta2", type = float, default = [0.999], nargs = '*')
	parser.add_argument("-tra", "--transport", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-tau", "--tau", type = float, default = [0], nargs = '*')
	parser.add_argument("-vta", "--varyingtau", type = int, default = [0], choices = [0, 1], nargs = '*')
	parser.add_argument("-lml", "--lambdaloss0", type = float, default = [1], nargs = '*')
	parser.add_argument("-lmt", "--lambdaloss0type", default = ['increasing'], choices = ['increasing', 'decreasing'], nargs = '*')
	parser.add_argument("-lmp", "--lambdaloss0power", type = float, default = [1], nargs = '*')
	parser.add_argument("-uzt", "--uzawatau", type = float, default = [0], nargs = '*')
	parser.add_argument("-uzs", "--uzawasteps", type = int, default = [0], nargs = '*')
	parser.add_argument("-ne0", "--nepochs0", type = int, default = [200], nargs = '*')
	parser.add_argument("-ne1", "--nepochs1", type = int, default = [50], nargs = '*')
	parser.add_argument("-ne2", "--nepochs2", type = int, default = [10], nargs = '*')
	parser.add_argument("-nro", "--nrounds", type = int, default = [5], nargs = '*')
	parser.add_argument("-trs", "--trainsize", type = float, default = [None], nargs = '*')
	parser.add_argument("-vls", "--valsize", type = float, default = [None], nargs = '*')
	parser.add_argument("-tss", "--testsize", type = float, default = [None], nargs = '*')
	parser.add_argument("-see", "--seed", type = int, default = [None], nargs = '*')
	parser.add_argument("-exp", "--experiments", action = 'store_true')
	parser.add_argument("-avg", "--averageexperiments", action = 'store_true')
	args = parser.parse_args()

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	if args.experiments or args.averageexperiments:
		parameters = [(name, values) for name, values in vars(args).items() if name not in ['experiments', 'averageexperiments']]
		experiments(parameters, args.averageexperiments)
	else :
		parameters = [values[0] for name, values in vars(args).items() if name not in ['experiments', 'averageexperiments']]
		experiment(*parameters, False)



