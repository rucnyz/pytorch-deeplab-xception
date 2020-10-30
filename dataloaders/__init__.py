import torch
from mypath import Path
from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader

import dataloaders.datasets.flood as flood


def make_data_loader(args, **kwargs):
	if args.dataset == 'pascal':
		train_set = pascal.VOCSegmentation(args, split = 'train')
		val_set = pascal.VOCSegmentation(args, split = 'val')
		if args.use_sbd:
			sbd_train = sbd.SBDSegmentation(args, split = ['train', 'val'])
			train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded = [val_set])

		num_class = train_set.NUM_CLASSES
		train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, **kwargs)
		val_loader = DataLoader(val_set, batch_size = args.batch_size, shuffle = False, **kwargs)
		test_loader = None

		return train_loader, val_loader, test_loader, num_class

	elif args.dataset == 'cityscapes':
		train_set = cityscapes.CityscapesSegmentation(args, split = 'train')
		val_set = cityscapes.CityscapesSegmentation(args, split = 'val')
		test_set = cityscapes.CityscapesSegmentation(args, split = 'test')
		num_class = train_set.NUM_CLASSES
		train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, **kwargs)
		val_loader = DataLoader(val_set, batch_size = args.batch_size, shuffle = False, **kwargs)
		test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle = False, **kwargs)

		return train_loader, val_loader, test_loader, num_class

	elif args.dataset == 'coco':
		train_set = coco.COCOSegmentation(args, split = 'train')
		val_set = coco.COCOSegmentation(args, split = 'val')
		num_class = train_set.NUM_CLASSES
		train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True, **kwargs)
		val_loader = DataLoader(val_set, batch_size = args.batch_size, shuffle = False, **kwargs)
		test_loader = None
		return train_loader, val_loader, test_loader, num_class
	elif args.dataset == 'flood':
		workpath = Path.db_root_dir('flood')
		train_data = flood.load_flood_train_data(workpath)
		train_dataset = flood.InMemoryDataset(train_data, flood.processAndAugment)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 16, shuffle = True, sampler = None,
		                                           batch_sampler = None, num_workers = 0, collate_fn = None,
		                                           pin_memory = True, drop_last = False, timeout = 0,
		                                           worker_init_fn = None)
		valid_data = flood.load_flood_valid_data(workpath)
		valid_dataset = flood.InMemoryDataset(valid_data, flood.processTestIm)
		valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 4, shuffle = True, sampler = None,
		                                           batch_sampler = None, num_workers = 0, collate_fn = lambda x: (
				torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
		                                           pin_memory = True, drop_last = False, timeout = 0,
		                                           worker_init_fn = None)
		test_data = flood.load_flood_valid_data(workpath)
		test_dataset = flood.InMemoryDataset(test_data, flood.processTestIm)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 4, shuffle = True, sampler = None,
		                                           batch_sampler = None, num_workers = 0, collate_fn = lambda x: (
				torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
		                                           pin_memory = True, drop_last = False, timeout = 0,
		                                           worker_init_fn = None)
		num_class = 2
		return train_loader, valid_loader, test_loader, num_class
	else:
		raise NotImplementedError
