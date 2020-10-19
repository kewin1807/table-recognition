import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
from constants import width_image, height_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder_structure, decoder_cell,  image_path, word_map_structure, word_map_cell, beam_size=3):
    k = beam_size
    vocab_size_structure = len(word_map_structure)
    vocab_size_cell = len(word_map_cell)
    img = imresize(img, (height_image, width_image))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

     # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_out = encoder(image)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

     # Flatten encoding
    # (1, num_pixels, encoder_dim)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    # (k, num_pixels, encoder_dim)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)