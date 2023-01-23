import torch.nn as nn, os, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.cm as cm, math
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from functools import partial
import torch.nn.functional as functional
from collections import OrderedDict

stack = lambda d :  {name: np.vstack(inp) for name, inp in d.items()}
get_avg = lambda d, n : [d[i].avg for i in range(n)]
l2norm = lambda x : np.sqrt(np.sum(x ** 2, axis = (1, 2, 3)))

convDiag = lambda x, M : functional.conv2d(x, M, stride = 1, padding = 1, groups = M.shape[0])
convDiagT = lambda x, M : functional.conv_transpose2d(x, M, stride = 1, padding = 1, groups = M.shape[0])

def test_autoencoder(datashape, encoder, decoder, testloader, mean, std):
	print('-' * 64, 'Testing autoencoder')
	print('-' * 64, 'Encoder\n', encoder)
	print('-' * 64, 'Decoder\n', decoder)
	criterion = nn.MSELoss()
	test_loss, idx_batch = 0, 4
	for i, (x, _) in enumerate(testloader):
		x = x.to(device)
		z = encoder(x)
		y = decoder(z)
		loss = criterion(y, x)
		test_loss += loss.item()
		if i == idx_batch:
			idx_images = np.random.choice(x.size()[0], 5, replace = False)
			x_ = x.cpu().detach().numpy().copy()[idx_images, :, :, :]
			y_ = y.cpu().detach().numpy().copy()[idx_images, :, :, :]
			show_autoencoder_images(x_, y_, mean, std, 'test-ae2.png')
			break
	test_loss /= (i + 1)
	print('-' * 64, 'Test loss : {:.4f}'.format(test_loss))
	print('-' * 64)

def topkaccuracy(output, target, topk = (1, )):
	maxk = max(topk)
	num = len(target)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.item() / num)
	return res

def create_autoencoder(inchannels = 3, filters = 100, ds = True, bn = True, imagenet = False, simpleencoder = False):
	intfilters = int(filters / 2)
	if imagenet:
		encoder = nn.Sequential(nn.Conv2d(3, filters, 7, 2, 3), nn.BatchNorm2d(filters), nn.ReLU(True), 
								nn.Conv2d(filters, filters, 3, 2, 1), nn.BatchNorm2d(filters), nn.ReLU(True))
		decoder = nn.Sequential(nn.ConvTranspose2d(filters, filters, 3, 2, 1, 1), nn.BatchNorm2d(filters), nn.ReLU(True), 
								nn.ConvTranspose2d(filters, 3, 7, 2, 3, 1), nn.BatchNorm2d(3), nn.Tanh())
		return encoder, decoder
	if simpleencoder:
		if ds:
			encoder = nn.Sequential(nn.Conv2d(inchannels, filters, 5, 2, 2), nn.BatchNorm2d(filters), nn.ReLU(True))
			decoder = nn.Sequential(nn.ConvTranspose2d(filters, inchannels, 5, 2, 2, 1), nn.BatchNorm2d(inchannels), nn.ReLU(True))
		else:
			encoder = nn.Sequential(nn.Conv2d(inchannels, filters, 3, 1, 1), nn.BatchNorm2d(filters), nn.ReLU(True))
			decoder = nn.Sequential(nn.Conv2d(filters, inchannels, 3, 1, 1), nn.BatchNorm2d(inchannels), nn.ReLU(True))
		return encoder, decoder
	if not ds:
		if not bn:
			encoder = nn.Sequential(nn.Conv2d(inchannels, intfilters, 3, 1, 1), nn.ReLU(True),
									nn.Conv2d(intfilters, filters, 3, 1, 1), nn.ReLU(True))
			decoder = nn.Sequential(nn.Conv2d(filters, intfilters, 3, 1, 1), nn.ReLU(True),
									nn.Conv2d(intfilters, inchannels, 3, 1, 1), nn.Tanh())
		else:
			encoder = nn.Sequential(nn.Conv2d(inchannels, intfilters, 3, 1, 1), nn.BatchNorm2d(intfilters), nn.ReLU(True),
									nn.Conv2d(intfilters, filters, 3, 1, 1), nn.BatchNorm2d(filters), nn.ReLU(True))
			decoder = nn.Sequential(nn.Conv2d(filters, intfilters, 3, 1, 1), nn.BatchNorm2d(intfilters), nn.ReLU(True),
									nn.Conv2d(intfilters, inchannels, 3, 1, 1), nn.BatchNorm2d(inchannels), nn.Tanh())
	else:
		if not bn:	
			encoder = nn.Sequential(nn.Conv2d(inchannels, intfilters, 3, 1, 1), nn.ReLU(True),
									nn.Conv2d(intfilters, filters, 5, 2, 2), nn.ReLU(True))
			decoder = nn.Sequential(nn.ConvTranspose2d(filters, intfilters, 5, 2, 2, 1), nn.ReLU(True),
									nn.ConvTranspose2d(intfilters, inchannels, 3, 1, 1), nn.Tanh())
		else:
			encoder = nn.Sequential(nn.Conv2d(inchannels, intfilters, 3, 1, 1), nn.BatchNorm2d(intfilters), nn.ReLU(True),
									nn.Conv2d(intfilters, filters, 5, 2, 2), nn.BatchNorm2d(filters), nn.ReLU(True))
			decoder = nn.Sequential(nn.ConvTranspose2d(filters, intfilters, 5, 2, 2, 1), nn.BatchNorm2d(intfilters), nn.ReLU(True),
									nn.ConvTranspose2d(intfilters, inchannels, 3, 1, 1), nn.BatchNorm2d(inchannels), nn.Tanh())
	return encoder, decoder

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()

class PredDecoder(Decoder):
	def __init__(self):
		super(Decoder, self).__init__()
	def forward(self, x):
		return self.layers(x)

class MLPSR(Decoder):
    def __init__(self, channels=256, size=32, classes=10, n_lin=3,
                 mlp_layers=3,  batchn=True, bias=True):
        super(MLPSR, self).__init__()
        self.n_lin=n_lin
        self.size=size

        self.init_pool = nn.AdaptiveAvgPool2d(math.ceil(self.size/4))
        self.blocks = []
        for n in range(self.n_lin):
            if batchn:
                bn_temp = nn.BatchNorm2d(channels)
            else:
                bn_temp = identity()

            conv = nn.Conv2d(channels, channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
            relu = nn.ReLU(True)
            self.blocks.append(nn.Sequential(conv,bn_temp,relu))

        self.blocks = nn.ModuleList(self.blocks)

        self.mlp_in_size = min(math.ceil(self.size/4), 2)
        self.out_pool = nn.AdaptiveAvgPool2d(self.mlp_in_size)
        mlp_feat = channels * (self.mlp_in_size) * (self.mlp_in_size)
        layers = []

        for l in range(mlp_layers):
            if l==0:
                in_feat = channels*self.mlp_in_size**2
                mlp_feat = mlp_feat
            else:
                in_feat = mlp_feat

            if batchn:
                bn_temp = nn.BatchNorm1d(mlp_feat)
            else:
                bn_temp = identity()

            layers +=[nn.Linear(in_feat, mlp_feat, bias=bias),
                            bn_temp, nn.ReLU(True)]
        layers += [nn.Linear(mlp_feat, classes, bias=bias)]
        self.classifier = nn.Sequential(*layers)
        self.num_layers = n_lin + mlp_layers + 1

    def forward(self, x):
        out = x
        out = self.init_pool(out)

        for n in range(self.n_lin):
            out = self.blocks[n](out)

        out = self.out_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]
		hidden_dim = int(round(inp * expand_ratio))
		self.use_res_connect = self.stride == 1 and inp == oup
		layers = []
		if expand_ratio != 1:
			layers.extend([nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias = False), nn.BatchNorm2d(hidden_dim), nn.ReLU(True)])
			layers.extend([nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = False), nn.BatchNorm2d(hidden_dim), nn.ReLU(True)])
		layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias = False))
		layers.append(nn.BatchNorm2d(oup))
		self.conv = nn.Sequential(*layers)
	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)



class PredConvLightDecoder(PredDecoder):
	def __init__(self, channels, size, classes, dim_in_decoder, expand_ratio=4, num_layers=1, final_dim=1024, bias=True):
		super(PredConvLightDecoder, self).__init__()
		self.in_planes = channels
		self.size = curr_size = size
		planes = dim_in_decoder
		layers = []
		channel_increment = planes // 2 if expand_ratio > 1 else planes
		layers.append(SepConvBNAct(self.in_planes, planes, stride = 2, bias = False))
		self.in_planes = planes
		curr_size = curr_size // 2

		out_planes = planes
		for i in range(num_layers-1):
			stride = 2 if curr_size >= 2 and i % 2 == 1 else 1
			out_planes = out_planes + channel_increment  if stride == 2 else out_planes

			layers.append(InvertedResidual(self.in_planes, out_planes,
										   expand_ratio=expand_ratio,
										   stride=stride))
			self.in_planes = out_planes
			curr_size = curr_size // 2 if stride == 2 else curr_size

		layers.append(ConvBNAct(self.in_planes, final_dim,
								kernel_size=1, padding=0, bias=False))
		self.in_planes = final_dim

		layers.append(nn.AdaptiveAvgPool2d(1))
		layers.append(View((-1, self.in_planes)))
		layers.append(nn.Linear(self.in_planes, classes, bias))
		self.num_layers = num_layers
		self.layers = nn.Sequential(*layers)

class View(nn.Module):
	def __init__(self, shape):
		super(View, self).__init__()
		self.shape = shape
	def forward(self, x):
		return x.view(*self.shape)

class ConvBNAct(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
		super(ConvBNAct, self).__init__()
		self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias), nn.BatchNorm2d(out_channels), nn.ReLU(True))
	def forward(self, x):
		return self.layers(x)

class SepConvBNAct(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False):
		super(SepConvBNAct, self).__init__()
		self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias = bias), nn.BatchNorm2d(out_channels), nn.ReLU(True), 
									nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, groups = out_channels, bias = bias), nn.BatchNorm2d(out_channels), nn.ReLU(True))
	def forward(self, x):
		return self.layers(x)


class MixedPredConvLightDecoder(PredDecoder):
	def __init__(self, channels, size, classes, bias = True):
		super(MixedPredConvLightDecoder, self).__init__()
		feat_mult = 4
		expand_ratio = 4
		self.possible_dim_in_decoders = [channels // expand_ratio]
		self.possible_num_layers = [i + 1 for i in range(0, 4)]
		self.decoders = nn.ModuleList()
		self.decoder_configs = []
		self.final_dim = 512 * feat_mult

		for dim_in_decoder, num_layer in product([self.possible_dim_in_decoders, self.possible_num_layers]):
			self.decoders.append(PredConvLightDecoder(channels, size, classes,
												 dim_in_decoder, num_layers=num_layer,
												 final_dim=self.final_dim,
												 expand_ratio = expand_ratio,
												 bias=bias))
			self.decoder_configs.append({
				'dim_in_decoder': dim_in_decoder,
				'num_layers': num_layer
			})

	@property
	def num_layers(self):
		'''Return maximum number of layers of all decoders'''
		num_layers = 0
		for decoder in self.decoders:
			num_layers = max(decoder.num_layers, num_layers)

		return num_layers

	@property
	def num_decoders(self):
		return len(self.decoders)

	def forward(self, x, weights=None, categorical=False):
		if weights is not None:
			if categorical:
				max_weight_idx = weights.max(0)[1].item()
				return self.decoders[max_weight_idx](x)*weights[max_weight_idx]
			else:
				return sum(weights[i] * self.decoders[i](x) for i in range(self.num_decoders))
		else:
			return sum(decoder(x) for decoder in self.decoders) / self.num_decoders

	

def create_classifier(name, nclasses, featureshape, apc = 0, bias = False):
	featuresize = int(np.prod(featureshape)) if (type(featureshape) is list or type(featureshape) is tuple) else featureshape
	if name == '1LIN':
		return nn.Sequential(nn.Flatten(), nn.Linear(featuresize, nclasses))
	if name == '2LIN':
		return nn.Sequential(nn.Flatten(), nn.Linear(featuresize, nclasses * 10), nn.Sigmoid(), nn.Linear(nclasses * 10, nclasses))
	if name == '3LIN':
		return nn.Sequential(nn.Flatten(), nn.Linear(featuresize, nclasses * 10), nn.BatchNorm1d(nclasses * 10), nn.ReLU(True), nn.Linear(nclasses * 10, nclasses))
	if name == '1CNN':
		nfilters = featureshape[1] if (type(featureshape) is list or type(featureshape) is tuple) else None
		if apc > 0:
			featuresize = int(nfilters * (featureshape[2] / apc) * (featureshape[3] / apc))
			return nn.Sequential(nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = bias), nn.ReLU(True), nn.AvgPool2d(apc), nn.BatchNorm2d(nfilters), nn.Flatten(), nn.Linear(featuresize, nclasses))
		else:
			return nn.Sequential(nn.Conv2d(nfilters, nfilters, 3, 1, 1, bias = bias), nn.ReLU(True), nn.BatchNorm2d(nfilters), nn.Flatten(), nn.Linear(featuresize, nclasses))
	if name == 'MPCL':
		return MixedPredConvLightDecoder(featureshape[1], featureshape[2], nclasses)
	if name == 'MLPS':
		return MLPSR(featureshape[1], featureshape[2], nclasses)
	if name == 'ALIN':
		return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(featureshape[1], nclasses))
	if name == "1C2F":
		nfilters = featureshape[1] if (type(featureshape) is list or type(featureshape) is tuple) else None
		intfilters = 32 if nfilters == 16 else (64 if nfilters < 128 else nfilters)
		stride = 2 if nfilters < 64 else 1
		return nn.Sequential(nn.Conv2d(nfilters, intfilters, 3, stride, 1, bias = bias), nn.BatchNorm2d(intfilters), nn.ReLU(True), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(intfilters, 128), nn.ReLU(True),
                    		 nn.Linear(128, nclasses))

def initialize(name, gain, module):
	if name == 'orthogonal':
		init = partial(nn.init.orthogonal_, gain = gain) 
	elif name == 'normal':
		init = partial(nn.init.normal_, mean = 0, std = gain) 
	elif name == 'kaiming':
		init = partial(nn.init.kaiming_normal_, a = 0, mode = 'fan_out', nonlinearity = 'relu')
	else:
		raise ValueError('Unknown init ' + name)
	if isinstance(module, nn.Conv2d):
		init(module.weight)
		if hasattr(module, 'bias') and module.bias is not None:
			nn.init.constant_(module.bias, 0)
	elif isinstance(module, nn.BatchNorm2d):
		if hasattr(module, 'weight') and module.weight is not None:
			nn.init.constant_(module.weight, 1)
		if hasattr(module, 'bias') and module.bias is not None:
			nn.init.constant_(module.bias, 0)
	elif isinstance(module, nn.Linear):
		init(module.weight)
		if hasattr(module, 'bias') and module.bias is not None:
			nn.init.constant_(module.bias, 0)

def product(iterables):
	if len(iterables) == 0 :
		yield ()
	else :
		it = iterables[0]
		for item in it :
			for items in product(iterables[1: ]) :
				yield (item, ) + items

def make_folder(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

class AverageMeter(object):
	def __init__(self):
		self.reset()
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	def update(self, val, num):
		self.val = val
		self.sum += val * num
		self.count += num
		self.avg = self.sum / self.count

def update_meters(y, pred, loss, loss_meter, acc_meter, trs = None, trs_meter = None, t = None, time_meter = None):
	num = len(y)
	correct = (pred == y).sum().item()
	accuracy = correct / num
	loss_meter.update(loss, num)
	acc_meter.update(accuracy, num)
	if trs is not None and trs_meter is not None:
		trs_meter.update(trs, num)
	if t is not None and time_meter is not None :
		time_meter.update(t, 1)

def plotscores(losses, accuracy, name):
	plt.figure(1)
	plt.subplot(211)
	plt.plot(losses)
	plt.ylabel('train loss')
	plt.subplot(212)
	plt.plot(accuracy)
	plt.xlabel('epoch')
	plt.ylabel('test accuracy')
	plt.savefig(name + '-loss-acc.png', bbox_inches = 'tight')
	plt.close()

def show_autoencoder_images(x, y, mean, std, name = None):
	bw = x.shape[1] == 1
	x = x[:, 0, :, :] if bw else np.moveaxis(x, 1, -1)
	y = y[:, 0, :, :] if bw else np.moveaxis(y, 1, -1)
	r, c = 2, x.shape[0]
	fig, axs = plt.subplots(r, c)
	for j in range(c):
		img = x[j, :, :] if bw else x[j, :, :, :]
		img = np.clip(std * img + mean, 0, 1)
		axs[0, j].imshow(img, cmap = 'gray' if bw else None)
		axs[0, j].axis('off')
		img = y[j, :, :] if bw else y[j, :, :, :]
		img = np.clip(std * img + mean, 0, 1)
		axs[1, j].imshow(img, cmap = 'gray' if bw else None)
		axs[1, j].axis('off')
	if name is not None:
		fig.savefig(name, bbox_inches = 'tight')
	else:
		plt.show()
	plt.close()

def show_decoded_images(images, mean, std, name = None):
	bw = images[0].shape[1] == 1
	rows, cols = images[0].shape[0], len(images)
	images = [img[:, 0, :, :] for img in images] if bw else [np.moveaxis(img, 1, -1) for img in images]
	col_names = ['og', 'ae'] + ['b{}'.format(i + 1) for i in range(cols - 2)]
	fig, axs = plt.subplots(rows, cols)
	for r in range(rows):
		for c in range(cols):
			img = images[c][r, :, :] if bw else images[c][r, :, :, :]
			img = np.clip(std * img + mean, 0, 1)
			axs[r, c].imshow(img, cmap = 'gray' if bw else None)
			axs[r, c].set_xticks([])
			axs[r, c].set_yticks([])
	for ax, col_name in zip(axs[0], col_names):
		ax.set_title(col_name)
	if name is not None:
		fig.savefig(name, bbox_inches = 'tight')
	else:
		plt.show()
	plt.close()

def plot_arrays(ratios, cosines, forcings, distances, nblocks, epoch, folder):
	plt.figure(figsize = (7, 7)) # plot cosine loss
	plt.plot(list(range(1, nblocks + 1)), cosines)
	plt.title('cos(F, grad L) after epoch ' + str(epoch))
	plt.xlabel('block')
	plt.ylabel('cos( F(h), grad_h L )')
	plt.savefig(os.path.join(folder, 'cos_epoch' + str(epoch) + '.png'), bbox_inches = 'tight')
	plt.close()
	plt.figure(figsize = (7, 7)) # plot forcing function and W2 movement
	plt.plot(list(range(1, nblocks + 1)), forcings, 'b', label = 'F(x)')
	if distances is not None:
		plt.plot(list(range(1, nblocks + 1)), distances, 'r', label = 'W2 movement')
	plt.title(('F and wasserstein distance' if distances is not None else 'F') + ' after epoch ' + str(epoch))
	plt.xlabel('block')
	plt.legend(loc = 'best')
	plt.savefig(os.path.join(folder, 'distance_epoch' + str(epoch) + '.png'), bbox_inches = 'tight')
	plt.close()
	plt.figure(figsize = (7, 7)) # plot cosine loss
	plt.plot(list(range(1, nblocks + 1)), ratios)
	plt.title('forcing function to input norm ratio after epoch ' + str(epoch))
	plt.xlabel('block')
	plt.ylabel('F(x) / x')
	plt.savefig(os.path.join(folder, 'ratio_epoch' + str(epoch) + '.png'), bbox_inches = 'tight')
	plt.close()

def convexHulls(points, labels):    
	convex_hulls = []
	for i in range(10):
		convex_hulls.append(ConvexHull(points[labels==i,:]))    
	return convex_hulls
	
def best_ellipses(points, labels):  
	gaussians = []    
	for i in range(10):
		gaussians.append(GaussianMixture(n_components=1, covariance_type='full').fit(points[labels==i, :])) 
	return gaussians
	
def Visualization(points2D, labels, convex_hulls, ellipses , nh, name = None, projname = 'tSNE'): 
	points2D_c = []
	for i in range(10):
		points2D_c.append(points2D[labels == i, :])
	cmap = cm.tab10 
	
	plt.figure(figsize=(3.841, 7.195), dpi = 100)
	plt.set_cmap(cmap)
	plt.subplots_adjust(hspace = 0.4)
	plt.subplot(311)
	plt.scatter(points2D[:, 0], points2D[:, 1], c = labels, s = 3, edgecolors = 'none', cmap = cmap, alpha = 1.0)# cmap=cm.Vega10cmap= , alpha=0.2)
	plt.colorbar(ticks = range(10))
	plt.title("2D " + projname + " - NH=" + str(nh*100.0))
	
	vals = [i / 10.0 for i in range(10)]
	sp2 = plt.subplot(312)
	for i in range(10):
		ch = np.append(convex_hulls[i].vertices,convex_hulls[i].vertices[0])
		sp2.plot(points2D_c[i][ch, 0], points2D_c[i][ch, 1], '-', label = '$%i$'%i, color = cmap(vals[i]))         
	plt.colorbar(ticks = range(10))
	plt.title(projname +" Convex Hulls")
	
	def plot_results(X, Y_, means, covariances, index, title, color):
		splot = ax
		for i, (mean, covar) in enumerate(zip(means, covariances)):
			v, w = linalg.eigh(covar)
			v = 2. * np.sqrt(2.) * np.sqrt(v)
			u = w[0] / linalg.norm(w[0])
			
			if not np.any(Y_ == i):
				continue
			plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color, alpha = 0.2)
	
			angle = np.arctan(u[1] / u[0])
			angle = 180. * angle / np.pi  
			ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
			ell.set_clip_box(splot.bbox)
			ell.set_alpha(0.6)
			splot.add_artist(ell)
	
		plt.title(title)
	
	ax = plt.subplot(3, 1, 3)
	for i in range(10):
		plot_results(points2D[labels==i, :], ellipses[i].predict(points2D[labels==i, :]), ellipses[i].means_, ellipses[i].covariances_, 0,projname+" fitting ellipses", cmap(vals[i]))
	if name is not None:
	  plt.savefig(name, bbox_inches = 'tight', pi = 100)
	else:
	  plt.show()
	plt.close()    
 
def neighboring_hit(points, labels):
	k = 6
	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
	distances, indices = nbrs.kneighbors(points)
	txs = 0.0
	txsc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	nppts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	for i in range(len(points)): 
		tx = 0.0
		for j in range(1,k+1):
			if labels[indices[i,j]]== labels[i]:
				tx += 1          
		tx /= k  
		txsc[labels[i]] += tx
		nppts[labels[i]] += 1
		txs += tx
	for i in range(10):
		txsc[i] /= nppts[i]
	return txs / len(points)






