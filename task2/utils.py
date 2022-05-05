import sys
import numpy
import os
import logging
import logging.handlers
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data.dataset import Dataset
import glob 

class CustomDataset(Dataset):
	def __init__(self, root):

		self.data_path = os.path.join(root, 'noisy')
		self.labels_path = os.path.join(root, 'clean')
		self.data_list = glob.glob(self.data_path+'/*/*')
		self.labels_list = glob.glob(self.labels_path+'/*/*')
		self.data_len = min(len(self.data_list),len(self.labels_list))

	def __len__(self):
		return self.data_len

	def __getitem__(self, idx):       
		item_path = self.data_list[idx]  
		path, file = os.path.split(item_path)

		data = np.load(item_path)
		data = torch.from_numpy(np.expand_dims(data, 0)).float()
		target_path = os.path.join(os.path.split(path)[1],file)
		label = np.load(os.path.join(self.labels_path, target_path))
		label = torch.from_numpy(np.expand_dims(label, 0)).float()

		return data, label


def collate_fun(batch):
	min_size = min([len(x[0][0]) for x in batch])
	data = torch.stack([item[0][0,0:min_size] for item in batch])
	target = torch.stack([item[1][0,0:min_size] for item in batch])	
	return torch.stack([data, target])

def train(epoch, net, train_loader, criterion, optimizer, device,  **kwargs):
	net.to(device)
	net.train()
	log_step = 50

	for batch_idx, (data, targets) in enumerate(train_loader):
		data, targets = data.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(data)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()


		if batch_idx % log_step == 0:
			print('Train Epoch {epoch} [{trained}/{total}]\tLoss: {:0.4f}\t'.format(
			loss.item(),
			epoch=epoch,
			trained=batch_idx*len(data),
			total=len(train_loader.dataset)
		))


def test(net, test_loader, criterion, device, **kwargs):
	net.eval()
	test_loss = 0.

	for batch_idx, (data, targets) in enumerate(test_loader):
		data, targets = data.to(device), targets.to(device)
		outputs = net(data)
		loss = criterion(outputs, targets)
		test_loss += loss.item()


	print('Test set: Average loss: {:.4f}'.format(
		test_loss /len(test_loader)
	))
	return test_loss


def get_logger(filename):
	logger = logging.getLogger("logger")
	logger.setLevel(logging.DEBUG)

	formatter = logging.Formatter(
		"[%(levelname)s | %(filename)s:%(lineno)s] %(asctime)s: %(message)s"
	)

	if not os.path.isdir("logs"):
		os.mkdir("logs")

	file_handler = logging.FileHandler("./logs/" + filename + ".log")
	stream_handler = logging.StreamHandler()

	file_handler.setFormatter(formatter)
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)

	return logger