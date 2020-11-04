import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import jsonlines
from bs4 import BeautifulSoup as bs
from html import escape
from constants import width_image, height_image


def create_input_files(image_folder="pubtabnet", output_folder="output",
                       max_len_token_structure=300,
                       max_len_token_cell=130, width_image=512,
                       height_image=512):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param json_file_path: path of Json data with splits, structure token, cell token, img_path
    :param image_folder: folder with downloaded images
    :param output_folder: folder to save files
    :param max_len_token_structure: don't sample captions_structure longer than this length
    :param max_len_token_cell: don't sample captions_structure longer than this length
    """

    # Read Karpathy JSON
    print("create_input .....")
    with open(os.path.join(image_folder, "PubTabNet_2.0.0.jsonl"), 'r') as reader:
        imgs = list(reader)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read image paths and captions for each image
    train_image_captions_structure = []
    train_image_captions_cells = []
    train_image_paths = []

    valid_image_captions_structure = []
    valid_image_captions_cells = []
    valid_image_paths = []

    test_image_captions_structure = []
    test_image_captions_cells = []
    test_image_paths = []
    word_freq_structure = Counter()
    word_freq_cells = Counter()

    max_number_imgs_train = 100000
    max_numver_imgs_val = 10000

    total_number_imgs_train = 0
    total_number_imgs_val = 0

    for (index, image) in enumerate(imgs):
        print("processing_image: {}".format(index))
        img = eval(image)
        word_freq_structure.update(img["html"]["structure"]["tokens"])

        for cell in img["html"]["cells"]:
            word_freq_cells.update(cell["tokens"])

        captions_structure = []
        caption_cells = []
        path = os.path.join("{}/{}".format(image_folder,
                                           img["split"]), img['filename'])
        if len(img["html"]["structure"]["tokens"]) <= max_len_token_structure:
            captions_structure.append(img["html"]["structure"]['tokens'])
            for cell in img["html"]["cells"]:
                caption_cells.append(cell["tokens"])

            if img["split"] == "train" and total_number_imgs_train <= max_number_imgs_train:
                train_image_captions_structure.append(captions_structure)
                train_image_captions_cells.append(caption_cells)
                train_image_paths.append(path)
                total_number_imgs_train += 1
            elif img["split"] == "val" and total_number_imgs_val <= max_numver_imgs_val:
                valid_image_captions_structure.append(captions_structure)
                valid_image_captions_cells.append(caption_cells)
                valid_image_paths.append(path)
                total_number_imgs_val += 1
            elif img["split"] == "test":
                test_image_captions_structure.append(captions_structure)
                test_image_captions_cells.append(caption_cells)
                test_image_paths.append(path)
            else:
                continue

    # create vocabluary structure
    words_structure = [w for w in word_freq_structure.keys()]
    word_map_structure = {k: v + 1 for v, k in enumerate(words_structure)}
    word_map_structure['<unk>'] = len(word_map_structure) + 1
    word_map_structure['<start>'] = len(word_map_structure) + 1
    word_map_structure['<end>'] = len(word_map_structure) + 1
    word_map_structure['<pad>'] = 0

    # create vocabluary cells
    words_cell = [w for w in word_freq_cells.keys()]
    word_map_cell = {k: v + 1 for v, k in enumerate(words_cell)}
    word_map_cell['<unk>'] = len(word_map_cell) + 1
    word_map_cell['<start>'] = len(word_map_cell) + 1
    word_map_cell['<end>'] = len(word_map_cell) + 1
    word_map_cell['<pad>'] = 0

    # save vocabluary to json
    with open(os.path.join(output_folder, 'WORDMAP_' + "STRUCTURE" + '.json'), 'w') as j:
        json.dump(word_map_structure, j)

    with open(os.path.join(output_folder, 'WORDMAP_' + "CELL" + '.json'), 'w') as j:
        json.dump(word_map_cell, j)

    # store image and encoding caption to h5 file
    for impaths, imcaps_structure, imcaps_cell, split in [(train_image_paths, train_image_captions_structure, train_image_captions_cells, 'train'),
                                                          (valid_image_paths, valid_image_captions_structure,
                                                           valid_image_captions_cells, 'val'),
                                                          (test_image_paths, test_image_captions_structure, test_image_captions_cells, 'test')]:
        if len(imcaps_structure) == 0:
            continue
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_.hdf5'), 'a') as h:
            images = h.create_dataset(
                'images', (len(impaths), 3, width_image, height_image), dtype='uint8')
            print("\nReading %s images and captions, storing to file...\n" % split)
            enc_captions_structure = []
            enc_captions_cells = []
            cap_structure_len = []
            cap_cell_len = []
            number_cell_per_images = []
            max_cells_per_images = max(
                [len(imcaps_cell[i]) for i in range(len(imcaps_cell))])
            for i, path in enumerate(tqdm(impaths)):
                captions_structure = imcaps_structure[i]
                captions_cell = imcaps_cell[i]
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(
                    img, (width_image, height_image), interp="cubic")
                img = img.transpose(2, 0, 1)
                # Save image to HDF5 file
                images[i] = img

                # encode caption cell and structure
                for j, c in enumerate(captions_structure):
                    enc_c = [word_map_structure['<start>']] + [word_map_structure.get(word, word_map_structure['<unk>']) for word in c] + [
                        word_map_structure['<end>']] + [word_map_structure['<pad>']] * (max_len_token_structure - len(c))
                    c_len = len(c) + 2
                    enc_captions_structure.append(enc_c)
                    cap_structure_len.append(c_len)
                # for each img have many cell captions
                each_enc_captions_cell = []
                each_cap_cell_len = []
                for j, c in enumerate(captions_cell):
                    enc_c = [word_map_cell['<start>']] + [word_map_cell.get(word, word_map_cell['<unk>']) for word in c] + [
                        word_map_cell['<end>']] + [word_map_cell['<pad>']] * (max_len_token_cell - len(c))
                    c_len = len(c) + 2
                    each_enc_captions_cell.append(enc_c)

                    each_cap_cell_len.append(c_len)
                # padding cell colection to (max_cells_per_images, max_len_token_cell+2)
                padding_enc_caption_cell = [
                    [0 for y in range(max_len_token_cell+2)] for x in range(max_cells_per_images - len(each_enc_captions_cell))]

                padding_len_caption_cell = [0 for x in range(
                    max_cells_per_images - len(each_enc_captions_cell))]

                each_enc_captions_cell += padding_enc_caption_cell
                each_cap_cell_len += padding_len_caption_cell

                # save encoding cell in per image
                enc_captions_cells.append(each_enc_captions_cell)
                cap_cell_len.append(each_cap_cell_len)
                number_cell_per_images.append(len(captions_cell))

            with open(os.path.join(output_folder, split + '_CAPTIONS_STRUCTURE' + '.json'), 'w') as j:
                json.dump(enc_captions_structure, j)
            with open(os.path.join(output_folder, split + '_CAPLENS_STRUCTURE' + '.json'), 'w') as j:
                json.dump(cap_structure_len, j)
            with open(os.path.join(output_folder, split + '_CAPTIONS_CELL' + '.json'), 'w') as j:
                json.dump(enc_captions_cells, j)
            with open(os.path.join(output_folder, split + '_CAPLENS_CELL' + '.json'), 'w') as j:
                json.dump(cap_cell_len, j)
            with open(os.path.join(output_folder, split + '_NUMBER_CELLS_PER_IMAGE' + '.json'), 'w') as j:
                json.dump(number_cell_per_images, j)


def id_to_word(vocabluary):
    id2word = {value: key for key, value in vocabluary.items()}
    return id2word


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(
            lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, encoder, decoder_structure, decoder_cell,
                    encoder_optimizer, decoder_structure_optimizer, decoder_cell_optimizer, recent_ted_score, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param recent_ted_score: validation TED score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'ted_score': recent_ted_score,
             'encoder': encoder,
             'decoder_structure': decoder_structure,
             'encoder_optimizer': encoder_optimizer,
             'decoder_structure_optimizer': decoder_structure_optimizer,
             'decoder_cell': decoder_cell,
             'decoder_cell_optimizer': decoder_cell_optimizer,
             }
    filename = 'checkpoint_table' + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

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


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def format_html(html):
    ''' Formats HTML code from tokenized annotation of img
    '''

    html_code = '''<html>
                   <head>
                   <meta charset="UTF-8">
                   <style>
                   table, th, td {
                     border: 1px solid black;
                     font-size: 10px;
                   }
                   </style>
                   </head>
                   <body>
                   <table frame="hsides" rules="groups" width="100%%">
                     %s
                   </table>
                   </body>
                   </html>''' % html

    # prettify the html
    soup = bs(html_code)
    html_code = soup.prettify()
    return html_code


def convertId2wordSentence(id2word, idwords):
    words = [id2word[idword] for idword in idwords]
    words = [word for word in words if word != "<end>" and word != "<start>"]
    words = "".join(words)
    return words
