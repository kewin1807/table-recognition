import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(
            data_folder, split + '_IMAGES_.hdf5'), 'r')
        self.imgs = self.h['images']

        # Load encoded captions structure
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_STRUCTURE' + '.json'), 'r') as j:
            self.captions_structure = json.load(j)

        # Load caption structure length (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_STRUCTURE' + '.json'), 'r') as j:
            self.caplens_structure = json.load(j)

        # Load encoded captions cell
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_CELL' + '.json'), 'r') as j:
            self.captions_cell = json.load(j)
        # Load caption cell length
        with open(os.path.join(data_folder, self.split + '_CAPLENS_CELL' + '.json'), 'r') as j:
            self.caplens_cell = json.load(j)
        with open(os.path.join(data_folder, self.split + "_NUMBER_CELLS_PER_IMAGE.json"), "r") as j:
            self.number_cell_per_images = json.load(j)

            # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of data image
        self.dataset_size = len(self.captions_structure)
        print(len(self.captions_structure) == len(self.captions_cell))

        print(len(self.caplens_cell) == len(self.caplens_structure))

        print(len(self.captions_structure) == len(self.caplens_structure))

    def __getitem__(self, i):
        # Remember, the Nth caption structure corresponds to the Nth image
        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption_structure = torch.LongTensor(self.captions_structure[i])

        caplen_structure = torch.LongTensor([self.caplens_structure[i]])

        print(np.array(self.captions_cell[i]).shape)
        captions_cell = torch.LongTensor(self.captions_cell[i])

        caplen_cell = torch.LongTensor(self.caplens_cell[i])

        number_cell_per_image = torch.LongTensor(
            [self.number_cell_per_images[i]])

        return img, caption_structure, caplen_structure, captions_cell, caplen_cell, number_cell_per_image

        # if self.split is 'TRAIN':
        #     return img, caption, caplen
        # else:
        #     # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
        #     all_captions = torch.LongTensor(
        #         self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
        #     return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size

# class CaptionCellPerImage(Dataset):
