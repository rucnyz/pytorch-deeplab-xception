# -*- coding: utf-8 -*-
# @Time    : 2020/10/29 23:58
# @Author  : nieyuzhou
# @File    : flood.py.py
# @Software: PyCharm
from time import time
import csv
import os
import numpy as np
import rasterio
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image


class InMemoryDataset(torch.utils.data.Dataset):

	def __init__(self, data_list, preprocess_func):
		self.data_list = data_list
		self.preprocess_func = preprocess_func

	def __getitem__(self, i):
		return self.preprocess_func(self.data_list[i])

	def __len__(self):
		return len(self.data_list)


def processAndAugment(data):
	(x, y) = data
	im, label = x.copy(), y.copy()

	# convert to PIL for easier transforms
	im1 = Image.fromarray(im[0])
	im2 = Image.fromarray(im[1])
	label = Image.fromarray(label.squeeze())

	# Get params for random transforms
	i, j, h, w = transforms.RandomCrop.get_params(im1, (256, 256))

	im1 = F.crop(im1, i, j, h, w)
	im2 = F.crop(im2, i, j, h, w)
	label = F.crop(label, i, j, h, w)
	if random.random() > 0.5:
		im1 = F.hflip(im1)
		im2 = F.hflip(im2)
		label = F.hflip(label)
	if random.random() > 0.5:
		im1 = F.vflip(im1)
		im2 = F.vflip(im2)
		label = F.vflip(label)

	norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
	im = torch.stack([transforms.ToTensor()(im1).squeeze(), transforms.ToTensor()(im2).squeeze()])
	im = norm(im)
	label = transforms.ToTensor()(label).squeeze()
	if torch.sum(label.gt(.003) * label.lt(.004)):
		label *= 255
	label = label.round()

	return im, label


def processTestIm(data):
	(x, y) = data
	im, label = x.copy(), y.copy()
	norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])

	# convert to PIL for easier transforms
	im_c1 = Image.fromarray(im[0]).resize((512, 512))
	im_c2 = Image.fromarray(im[1]).resize((512, 512))
	label = Image.fromarray(label.squeeze()).resize((512, 512))

	im_c1s = [F.crop(im_c1, 0, 0, 256, 256), F.crop(im_c1, 0, 256, 256, 256),
	          F.crop(im_c1, 256, 0, 256, 256), F.crop(im_c1, 256, 256, 256, 256)]
	im_c2s = [F.crop(im_c2, 0, 0, 256, 256), F.crop(im_c2, 0, 256, 256, 256),
	          F.crop(im_c2, 256, 0, 256, 256), F.crop(im_c2, 256, 256, 256, 256)]
	labels = [F.crop(label, 0, 0, 256, 256), F.crop(label, 0, 256, 256, 256),
	          F.crop(label, 256, 0, 256, 256), F.crop(label, 256, 256, 256, 256)]

	ims = [torch.stack((transforms.ToTensor()(x).squeeze(),
	                    transforms.ToTensor()(y).squeeze()))
	       for (x, y) in zip(im_c1s, im_c2s)]

	ims = [norm(im) for im in ims]
	ims = torch.stack(ims)

	labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
	labels = torch.stack(labels)

	if torch.sum(labels.gt(.003) * labels.lt(.004)):
		labels *= 255
	labels = labels.round()

	return ims, labels


def getArrFlood(fname):
	return rasterio.open(fname).read()


def download_flood_water_data_from_list(l):
	i = 0
	tot_nan = 0
	tot_good = 0
	flood_data = []
	for (im_fname, mask_fname) in l:
		if not os.path.exists(os.path.join("files/", im_fname)):
			continue
		arr_x = np.nan_to_num(getArrFlood(os.path.join("files/", im_fname)))
		arr_y = getArrFlood(os.path.join("files/", mask_fname))
		ignore = (arr_y == -1)
		ignore = ((np.uint8(ignore) * -1) * 256) + 1
		arr_y *= ignore
		arr_y = np.uint8(getArrFlood(os.path.join("files/", mask_fname)))
		if np.sum((arr_y != arr_y)) == 0:
			arr_x = np.clip(arr_x, -50, 1)
			arr_x = (arr_x + 50) / 51

			if i % 100 == 0:
				print(im_fname, mask_fname)
			i += 1
			flood_data.append((arr_x, arr_y))
		else:
			print("skipping nan")

	return flood_data


def load_flood_train_data(workpath):
	fname = workpath + "flood_train_data.csv"
	training_files = []
	with open(fname) as f:
		for line in csv.reader(f):
			training_files.append(tuple((workpath + line[0], workpath + line[1])))

	return download_flood_water_data_from_list(training_files)


def load_flood_valid_data(workpath):
	fname = workpath + "flood_valid_data.csv"
	validation_files = []
	with open(fname) as f:
		for line in csv.reader(f):
			validation_files.append(tuple((workpath + line[0], workpath + line[1])))

	return download_flood_water_data_from_list(validation_files)


def load_flood_test_data(workpath):
	fname = workpath + "flood_test_data.csv"
	testing_files = []
	with open(fname) as f:
		for line in csv.reader(f):
			testing_files.append(tuple((workpath + line[0], workpath + line[1])))

	return download_flood_water_data_from_list(testing_files)
