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
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encoderImage(encoder, image_path):
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
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
    return encoder_out


def structure_image_beam_search(encoder_out, decoder, word_map, beam_size=3):
    k = beam_size
    vocab_size = len(word_map)
    decoder_structure_dim = 256
    # Read image and process

    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    # (1, num_pixels, encoder_dim)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    # (k, num_pixels, encoder_dim)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step decode structure; construct just <start>
    k_prev_words = torch.LongTensor(
        [[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas;
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

    # tensor save hidden state and after filter to choice hidden state to pass cell decoder
    seqs_hidden_states = torch.zeros(k, 1, decoder_structure_dim).to(device)

    # Lists to store completed sequences, their alphas and scores, hidden
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()
    complete_seqs_hiddens = list()

    # start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(
            k_prev_words).squeeze(1)  # (s, embed_dim)

        # (s, encoder_dim), (s, num_pixels)
        awe, alpha = decoder.attention(encoder_out, h)
        # (s, enc_image_size, enc_image_size)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)

        # gating scalar, (s, encoder_dim)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        h, c = decoder.decode_step(
            torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas, and hidden_state
        seqs = torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        if step == 1:
            seqs_hidden_states = h.unsqueeze(1)
        else:
            seqs_hidden_states = torch.cat(
                [seqs_hidden_states[prev_word_inds], h[prev_word_inds].unsqueeze(1)], dim=1)  # (s, step+1, decoder_structure_dim)
            # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(
            set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            complete_seqs_hiddens.extend(
                seqs_hidden_states[complete_inds].tolist())
        k -= len(complete_inds)  # reduce beam length accordingly

        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        seqs_hidden_states = seqs_hidden_states[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]

        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        # if step > 50:
        #     break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    hidden_states = complete_seqs_hiddens[i]

    return seq, alphas, hidden_states


def cell_image_beam_search(encoder_out, decoder, word_map, hidden_state_structure, beam_size=3):
    k = beam_size
    vocab_size = len(word_map)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)
    decoder_structure_dim = 256

    # Flatten encoding
    # (1, num_pixels, encoder_dim)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    # (k, num_pixels, encoder_dim)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step decode structure; construct just <start>
    k_prev_words = torch.LongTensor(
        [[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas;
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(
            k_prev_words).squeeze(1)  # (s, embed_dim)

        # (s, encoder_dim), (s, num_pixels)
        awe, alpha = decoder.attention(encoder_out, h)
        # (s, enc_image_size, enc_image_size)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)

        # gating scalar, (s, encoder_dim)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        s = list(awe.size())[0]
        # convert list to tensor
        hidden_state_structure = torch.stack(hidden_state_structure)
        hidden_state_structure = hidden_state_structure.expand(
            s, decoder_structure_dim)
        # concat hidden state to attention and decode
        awe = torch.cat(
            (awe, hidden_state_structure), dim=1)

        h, c = decoder.decode_step(
            torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas, and hidden_state
        seqs = torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(
            set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        seqs_hidden_states = seqs_hidden_states[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]

        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        # if step > 50:
        #     break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    return seq, alpha


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        # if t > 50:
        #     break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black',
                 backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(
                current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(
                current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map_structure', '-wms',
                        help='path to word map structure JSON')
    parser.add_argument('--word_map_cell', '-wmc',
                        help='path to word map cell JSON')
    parser.add_argument('--beam_size_structure', '-bs', default=3,
                        type=int, help='beam size for beam search')
    parser.add_argument('--beam_size_cell', '-bc', default=3,
                        type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth',
                        action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder_structure = checkpoint['decoder_structure']
    decoder_structure = decoder_structure.to(device)
    decoder_structure.eval()

    decoder_cell = checkpoint["decoder_cell"]
    decoder_cell = decoder_cell.to(device)
    decoder_cell.eval()

    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    with open(args.word_map_structure, 'r') as j:
        word_map_structure = json.load(j)
    with open(args.word_map_cell, "r") as j:
        word_map_cell = json.load(j)
    id2word_stucture = id_to_word(word_map_structure)
    id2word_cell = id_to_word(word_map_cell)

    encoder_out = encoderImage(encoder, args.img)

    seq, alphas, hidden_states = structure_image_beam_search(
        encoder_out, decoder_structure, word_map_structure, beam_size=args.beam_size_structure)

    cells = []
    html = ""
    for index, s in seq:
        html += id2word_stucture[str(s)]
        if id2word_stucture[str(s)] == "<td>" or id2word_stucture[str(s)] == ">":
            hidden_state_structure = hidden_states[index]
            seq_cell, alphas = cell_image_beam_search(
                encoder_out, decoder_cell, word_map_cell, hidden_state_structure, beam_size=args.beam_size_cell)

            html_cell = convertId2wordSentence(id2word_cell, seq_cell)
            html += html_cell

    print(html)
