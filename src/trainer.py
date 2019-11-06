from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import pickle
import os.path as osp
import numpy as np

import torch
import torch.onnx
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import src.models as models
import src.data_manager as data_manager
from src.data_manager.samplers import RandomPertSampler
from src.optimizer import init_optim
from src.utils.torchtools import count_num_param, calc_binary_accuracy
from src.utils.iotools import save_checkpoint, check_isfile
from src.utils.logger import Logger
from src.utils.metrics.averagemeter import AverageMeter
from src.utils.metrics.rocmeter import RocMeter
from src.utils.metrics.mapmeter import mAPMeter
from distutils.dir_util import copy_tree

from src.loss.focalloss import FocalLoss
from src.loss.diceloss import dice_loss
from src.loss.binary_cross_entropy import weighted_binary_cross_entropy


parser = argparse.ArgumentParser(description='DeepSide TRAIN')
# Dataset options
parser.add_argument('--data', default="Multiple", choices=data_manager.get_names(),
					help="Dataset function")
parser.add_argument('--data-type', default="GE_CS_META",
					help="Data type")
parser.add_argument('--multimodal', default=False,
					help="Generate dataset compatible with multimodal learning")
parser.add_argument('--n-fold', default=3,
					help="n fold cross validation")
parser.add_argument('--use-sampler', default=False,
					help="Random perturbation sampler")
parser.add_argument('--add-dmso', default=False,
					help="Add dmso ge values to train set")
parser.add_argument('--dmso-norm', default=False,
					help="DMSO normalized dataset")
parser.add_argument('--concat-dmso', default=False,
					help="merge drug GE and DMSO GE samples")
parser.add_argument('--train-signatures', default=True,
					help='Train with only strongest experiments')
parser.add_argument('--test-signatures', default=True,
					help='Test with only strongest experiments')
parser.add_argument('--parent-labels', default=True)

# Network options
parser.add_argument('--cs-hidden-layers', default=[2000, 2000, 2000],
					help="hidden layer neuron sizes for cs network")
parser.add_argument('--ge-hidden-layers', default=[2000, 2000, 2000],
					help="hidden layer neuron sizes for ge&cs network")
parser.add_argument('--meta-hidden-layers', default=[2000, 2000, 2000],
					help="hidden layer neuron sizes for ge&cs network")
parser.add_argument('--embedding-size', default=2000,
					help="embedding feature dimension")
parser.add_argument('--cls-layers', default=[2000, 2000],
					help="classification net neuron sizes of the MMAttention")
parser.add_argument('--hidden-layers', default=[8000, 8000],
					help="hidden layer neuron sizes MLP")
parser.add_argument('--res-in-out', default=2000,
					help="residual block input and output layers size")
parser.add_argument('--res-middle', default=2000,
					help="residual block middle layer size")
parser.add_argument('--res-blocks', default=2,
					help="number of residual blocks")
parser.add_argument('--arch', default="mlpsidenet", choices=models.get_names(),
					help="Model Name")
parser.add_argument('--dropout', default=0.2, type=float)

# Optimization options
parser.add_argument('--optim', type=str, default='adam',
					help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=500, type=int,
					help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
					help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=64, type=int,
					help="train batch size")
parser.add_argument('--test-batch', default=24, type=int,
					help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
					help="initial learning rate")
parser.add_argument('--stepsize', default=[50, 300], nargs='+', type=int,
# parser.add_argument('--stepsize', default=[200, 300, 400], nargs='+', type=int,
					help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
					help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-06, type=float,
					help="weight decay (default: 5e-04)")
parser.add_argument('--label-smooth', action='store_true',
					help="use label smoothing regularizer in cross entropy loss")

# Miscs
parser.add_argument('--print-freq', type=int, default=3,
					help="print frequency")
parser.add_argument('--seed', type=int, default=1,
					help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--load-weights', type=str, default='',
					help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--evaluate', action='store_true',
					help="evaluation only")
parser.add_argument('--eval-step', type=int, default=2,
					help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-dir', type=str, default='./../log')
parser.add_argument('--use-cpu', action='store_true',
					help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str,
					help='gpu device ids for CUDA_VISIBLE_DEVICES')

# global variables
args = parser.parse_args()


def main(trainloader, testloader, use_gpu, dataset_config):
	best_auc = -np.inf

	print("Initializing LINCS dataset")
	train_dataset = trainloader.dataset
	test_dataset = testloader.dataset
	
	print("===> Train Dataset <===")
	train_dataset._print()
	print()
	print("===> Test Dataset <===")
	test_dataset._print()
	print()

	train_history = dict()
	train_history["losses"] = []
	train_history["accuracies"] = []
	
	test_history = dict()
	test_history["accuracies"] = []
	test_history["map_scores"] = []
	test_history["roc_scores"] = []
	
	print("Initializing DrugNet model")

	model_config = {	"name": args.arch,
                        "p_dropout": args.dropout,
# 	Multimodal configuration
						"ge_in": train_dataset.feature_ge,
						"ge_layers": args.ge_hidden_layers,
						"cs_in": train_dataset.feature_cs,
						"cs_layers": args.cs_hidden_layers,
						"meta_in": train_dataset.feature_meta,
						"meta_layers": args.meta_hidden_layers,
						"feature_dim": args.embedding_size,
						"cls_layers": args.cls_layers,
#	MLP configuration
						"in_dim": train_dataset.feature_size,
						"out_dim": train_dataset.output_size,
						"hid_layers": args.hidden_layers,
#	ResMLP configuration
						"res_dim": args.res_in_out,
						"hid_dim": args.res_middle,
						"n_res_block": args.res_blocks
					}
	model = models.init_model(**model_config)
	
	print("Model size: {:.3f} M".format(count_num_param(model)))

	criterions = []
	# criterions.append(nn.BCELoss())
	criterions.append(nn.BCEWithLogitsLoss())
	# criterions.append(FocalLoss(logits=False, gamma=2))
	# criterions.append(weighted_binary_cross_entropy)
	print(criterions)

	optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

	if args.load_weights and check_isfile(args.load_weights):
		# load pretrained weights but ignore layers that don't match in size
		checkpoint = torch.load(args.load_weights)
		pretrain_dict = checkpoint['state_dict']
		model_dict = model.state_dict()
		pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
		model_dict.update(pretrain_dict)
		model.load_state_dict(model_dict)
		print("Loaded pretrained weights from '{}'".format(args.load_weights))

	if args.resume and check_isfile(args.resume):
		checkpoint = torch.load(args.resume)
		model.load_state_dict(checkpoint['state_dict'])
		args.start_epoch = checkpoint['epoch'] + 1
		best_rank1 = checkpoint['rank1']
		print("Loaded checkpoint from '{}'".format(args.resume))
		print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, best_rank1))

	if use_gpu:
		model = nn.DataParallel(model).cuda()

	if args.evaluate:
		print("Evaluate only")
		auc = test(model, testloader, test_history, use_gpu)
		return

	start_time = time.time()
	train_time = 0
	best_epoch = args.start_epoch
	print("==> Start training")

	for epoch in range(args.start_epoch, args.max_epoch):
		start_train_time = time.time()
		train(epoch, model, criterions, optimizer, trainloader, train_history, use_gpu)
		train_time += round(time.time() - start_train_time)
		
		scheduler.step()
		
		if args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
			print("==> Test")
			auc = test(model, criterions, testloader, test_history, use_gpu)
			is_best = auc > best_auc

			if is_best:
				best_auc = auc
				best_epoch = epoch + 1
			
			if use_gpu:
				state_dict = model.module.state_dict()
			else:
				state_dict = model.state_dict()
			
			save_checkpoint({
				'state_dict': state_dict,
				'auc': auc,
				'epoch': epoch,
				'model_config': model_config,
				'dataset_config': dataset_config,
				'optim': args.optim,
				'lr': args.lr,
				'batch': args.train_batch,
			}, is_best, args.save_dir)

	print("==> Best Loss {:.1%}, achieved at epoch {}".format(best_auc, best_epoch))

	elapsed = round(time.time() - start_time)
	elapsed = str(datetime.timedelta(seconds=elapsed))
	train_time = str(datetime.timedelta(seconds=train_time))
	print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
	return train_history, test_history, best_auc


def train(epoch, model, criterions, optimizer, trainloader, history, use_gpu):
	losses = AverageMeter()
	batch_time = AverageMeter()
	data_time = AverageMeter()
	accuracies = AverageMeter()

	model.train()

	end = time.time()
	for batch_idx, (data, labels) in enumerate(trainloader):
		data_time.add(time.time() - end)

		if type(data) is list:
			if len(data) == 2:
				data_ge, data_cs = data
				if use_gpu:
					data_ge = data_ge.cuda()
					data_cs = data_cs.cuda()
					labels = labels.cuda()
				outputs = model(data_ge, data_cs)
			elif len(data) == 3:
				data_ge, data_cs, data_meta = data
				if use_gpu:
					data_ge = data_ge.cuda()
					data_cs = data_cs.cuda()
					data_meta = data_meta.cuda()
					labels = labels.cuda()
				outputs = model(data_ge, data_cs, data_meta)
		else:
			if use_gpu:
				data, labels = data.cuda(), labels.cuda(non_blocking=True)
			if args.arch == 'mlpontology':
				ontology_vectors = trainloader.dataset.ontology_vectors
				ontology_vectors = torch.from_numpy(ontology_vectors).type(torch.FloatTensor)
				ontology_vectors = ontology_vectors.cuda()
				outputs = model(data, ontology_vectors)
			else:
				outputs = model(data)

		loss = 0
		if isinstance(outputs, tuple):
			for out_i, output in enumerate(outputs):
				out_w = 1
				if out_i == 0 and len(outputs) != 1:
					out_w = 2
				for criterion in criterions:
					if criterion is weighted_binary_cross_entropy:
						loss += criterion(input=output, target=labels, class_weights=trainloader.dataset.class_weights) * out_w
					else:
						loss += criterion(input=output, target=labels) * out_w
			outputs = outputs[0]
		else:
			for criterion in criterions:
				if criterion is weighted_binary_cross_entropy:
					loss += criterion(input=outputs, target=labels, class_weights=trainloader.dataset.class_weights)
				else:
					loss += criterion(input=outputs, target=labels)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		acc = calc_binary_accuracy(outputs, labels)
		batch_time.add(time.time() - end)
		losses.add(loss.item(), labels.size(0))
		accuracies.add(acc, labels.size(0))

		if (batch_idx + 1) % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accu {acc.val:.4f} ({acc.avg:.4f})\t'.format(
				   epoch + 1, batch_idx + 1, len(trainloader), 
				   batch_time=batch_time,
				   data_time=data_time, 
				   loss=losses,
				   acc=accuracies))
		
		end = time.time()
		history["losses"].append(losses.avg)
		history["accuracies"].append(accuracies.avg)


def test(model, criterions, testloader, history, use_gpu):
	losses = AverageMeter()
	batch_time = AverageMeter()
	accuracies = AverageMeter()
	map_scores = mAPMeter()
	roc_scores = RocMeter()
	np_outputs = []
	np_targets = []

	model.eval()

	with torch.no_grad():
		end = time.time()
		for batch_idx, (data, labels) in enumerate(testloader):
			if type(data) is list:
				if len(data) == 2:
					data_ge, data_cs = data
					if use_gpu:
						data_ge = data_ge.cuda()
						data_cs = data_cs.cuda()
						labels = labels.cuda()
					outputs = model(data_ge, data_cs)
				elif len(data) == 3:
					data_ge, data_cs, data_meta = data
					if use_gpu:
						data_ge = data_ge.cuda()
						data_cs = data_cs.cuda()
						data_meta = data_meta.cuda()
						labels = labels.cuda()
					outputs = model(data_ge, data_cs, data_meta)
			else:
				if use_gpu:
					data, labels = data.cuda(), labels.cuda(non_blocking=True)

				if args.arch == 'mlpontology':
					ontology_vectors = trainloader.dataset.ontology_vectors
					ontology_vectors = torch.from_numpy(ontology_vectors).type(torch.FloatTensor)
					ontology_vectors = ontology_vectors.cuda()
					outputs = model(data, ontology_vectors)
				else:
					outputs = model(data)

			loss = 0
			if isinstance(outputs, tuple):
				for out_i, output in enumerate(outputs):
					out_w = 1
					if out_i == 0 and len(outputs) != 1:
						out_w = 2
					for criterion in criterions:
						if criterion is weighted_binary_cross_entropy:
							loss += criterion(input=output, target=labels,
											  class_weights=trainloader.dataset.class_weights) * out_w
						else:
							loss += criterion(input=output, target=labels) * out_w
				outputs = outputs[0]
			else:
				for criterion in criterions:
					if criterion is weighted_binary_cross_entropy:
						loss += criterion(input=outputs, target=labels, class_weights=trainloader.dataset.class_weights)
					else:
						loss += criterion(input=outputs, target=labels)

			acc = calc_binary_accuracy(outputs, labels)

			losses.add(loss.item(), labels.size(0))
			batch_time.add(time.time() - end)
			accuracies.add(acc, labels.size(0))
			np_outputs.append(outputs.cpu().data)
			np_targets.append(labels.cpu().data)
			# map_scores.add(outputs.data, labels.data)
			# roc_scores.add(outputs.data, labels.data)

		map_scores.add(torch.cat(np_outputs, 0), torch.cat(np_targets, 0))
		roc_scores.add(torch.cat(np_outputs, 0), torch.cat(np_targets, 0))

		print('Time {batch_time.avg:.3f}\t'
			  'Acc {acc.avg:.4f}\t'
			  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
			  'mAP {mAP:.4f}\t'
			  'AUC {auc:.4f}'.format(
			   batch_time=batch_time,
			   loss=losses,
			   acc=accuracies,
			   mAP=map_scores.value(),
			   auc=roc_scores.value()))
		
		end = time.time()
	
	history["map_scores"].append(map_scores.value())
	history["roc_scores"].append(roc_scores.value())
	history["accuracies"].append(accuracies.avg)
	return roc_scores.value()
	# return losses.avg


if __name__ == '__main__':
	date_str = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
	args.save_dir = osp.join(args.save_dir, args.arch, (args.data_type + "_" + date_str))
	base_save_dir = args.save_dir
	torch.manual_seed(args.seed)
	use_gpu = torch.cuda.is_available()
	if args.use_cpu: use_gpu = False

	copy_tree("../src", osp.join(args.save_dir, "src"))
	
	if not args.evaluate:
		sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
	else:
		sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
	print("==========\nArgs:{}\n==========".format(args))

	if use_gpu:
		print("Currently using GPU {}".format(args.gpu_devices))
		cudnn.benchmark = True
		torch.cuda.manual_seed_all(args.seed)
	else:
		print("Currently using CPU (GPU is highly recommended)")

	dataset_config = { 'name': args.data,
					   'n_fold': args.n_fold,
					   'data_type': args.data_type,
					   'multimodal': args.multimodal,
					   'use_sampler': args.use_sampler,
					   'add_dmso': args.add_dmso,
					   'dmso_norm': args.dmso_norm,
					   'concat_dmso': args.concat_dmso,
                       'train_signatures': args.train_signatures,
                       'test_signatures': args.test_signatures,
					   'parent_labels': args.parent_labels
					   }
	dataset = data_manager.init_dataset(**dataset_config)
	train_fold_history = []
	test_fold_history = []
	best_auc_scores = []
	
	for fold_i, (train_dataset, test_dataset) in enumerate(dataset):
		args.save_dir = os.path.join(base_save_dir, "fold"+str(fold_i))
		if args.use_sampler:
			trainloader = DataLoader(
				train_dataset,
				batch_size=args.train_batch,
				sampler=RandomPertSampler(train_dataset), 
				drop_last=True
			)
		else:
			trainloader = DataLoader(
				train_dataset,
				batch_size=args.train_batch,
				shuffle=True, 
				drop_last=True,
				num_workers=4,
				pin_memory=True
			)
	
		testloader = DataLoader(
			test_dataset,
			batch_size=args.test_batch,
			shuffle=False, 
			drop_last=False,
			num_workers=4,
			pin_memory=True
		)
		
		train_hist, test_hist, best_auc = main(trainloader, testloader, use_gpu, dataset_config)
		train_fold_history.append(train_hist)
		test_fold_history.append(test_hist)
		best_auc_scores.append(best_auc)
	
	hist = { 'train': train_fold_history,
			 'test': test_fold_history}
	with open(osp.join(base_save_dir, "history.pickle"), 'wb') as handle:
		pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)