import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderStuctureWithAttention, DecoderCellPerImageWithAttention
from dataset import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

data_folder = "output"

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# training_parameter
start_epoch = 0
epochs = 120

# keeps track of number of epochs since there's been an improvement in validation BLEU
epochs_since_improvement = 0
batch_size = 32
batch_size_cell_per_image = 4

workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none


def main():
    global checkpoint
    word_map_structure_file = os.path.join(
        data_folder, "WORDMAP_STRUCTURE.json")
    word_map_cell_file = os.path.join(data_folder, "WORDMAP_CELL.json")

    with open(word_map_structure_file, "r") as j:
        word_map_structure = json.load(j)
    with open(word_map_cell_file, "r") as j:
        word_map_cell = json.load(j)

    if checkpoint is None:
        decoder_structure = DecoderStuctureWithAttention(attention_dim=attention_dim,
                                                         embed_dim=emb_dim,
                                                         decoder_dim=decoder_dim,
                                                         vocab_size=len(
                                                             word_map_structure),
                                                         dropout=dropout)
        decoder_cell = DecoderCellPerImageWithAttention(
            attention_dim=attention_dim, embed_dim=emb_dim, decoder_dim=decoder_dim, vocab_size=len(word_map_cell), dropout=0.2)
        decoder_structure_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_structure.parameters()),
                                                       lr=decoder_lr)
        decoder_cell_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_structure.parameters()),
                                                  lr=decoder_lr)

        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        decoder_structure = checkpoint['decoder_structure']
        decoder_structure_optimizer = checkpoint["decoder_structure_optimizer"]

        decoder_cell = checkpoint["decoder_cell"]
        decoder_cell_optimizer = checkpoint["decoder_cell_optimizer"]

        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']

        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder_structure = decoder_structure.to(device)
    decoder_cell = decoder_cell.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'train',
                       transform=transforms.Compose([normalize])), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'val',
                       transform=transforms.Compose([normalize])), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # train foreach epoch
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_structure, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train(train_loader=train_loader,
              encoder=encoder,
              decoder_structure=decoder_structure,
              decoder_cell=decoder_cell,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_structure_optimizer=decoder_structure_optimizer,
              decoder_cell_optimizer=decoder_cell_optimizer,
              epoch=epoch)


def train(train_loader, encoder, decoder_structure, decoder_cell, criterion, encoder_optimizer, decoder_structure_optimizer, decoder_cell_optimizer):

    decoder_structure.train()
    decoder_cell.train()
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    for i, (imgs, caption_structures, caplen_structures, caption_cells, caplen_cells, number_cell_per_images) in enumerate(train_loader):
        imgs = imgs.to(device)
        caption_structures = caption_structures.to(device)
        caplen_structures = caplen_structures.to(device)

        # Foward encoder image and decoder structure
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder_structure(
            imgs, caption_structures, caplen_structures)
