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
from metric.metric_score import TEDS
import numpy as np

data_folder = "output"

# Model parameters
emb_dim_structure = 16  # dimension of word embeddings
emb_dim_cell = 80
attention_dim = 512  # dimension of attention linear layers
decoder_dim_structure = 256  # dimension of decoder RNN structure
decoder_dim_cell = 512

dropout = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# training_parameter
start_epoch = 0
epochs = 120

# keeps track of number of epochs since there's been an improvement in validation BLEU
epochs_since_improvement = 0
batch_size = 2

workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-3  # learning rate for encoder if fine-tuning
decoder_lr = 4e-3  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_TED = 0.  # TED score right now
print_freq = 10  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none
hyper_loss = 0.5
word_map_structure_file = os.path.join(
    data_folder, "WORDMAP_STRUCTURE.json")
word_map_cell_file = os.path.join(data_folder, "WORDMAP_CELL.json")
teds = TEDS(n_jobs=4)

with open(word_map_structure_file, "r") as j:
    word_map_structure = json.load(j)
with open(word_map_cell_file, "r") as j:
    word_map_cell = json.load(j)
id2word_stucture = id_to_word(word_map_structure)
id2word_cell = id_to_word(word_map_cell)


def main():
    global checkpoint, start_epoch, fine_tune_encoder, word_map_structure, word_map_cell, epochs_since_improvement, hyper_loss, id2word_stucture, id2word_cell, teds, best_TED

    if checkpoint is None:
        decoder_structure = DecoderStuctureWithAttention(attention_dim=attention_dim,
                                                         embed_dim=emb_dim_structure,
                                                         decoder_dim=decoder_dim_structure,
                                                         vocab=word_map_structure,
                                                         dropout=dropout)
        decoder_cell = DecoderCellPerImageWithAttention(
            attention_dim=attention_dim, embed_dim=emb_dim_cell, decoder_dim=decoder_dim_cell, vocab_size=len(word_map_cell), dropout=0.2, decoder_structure_dim=decoder_dim_structure)
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
        best_TED = checkpoint['ted_score']

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

    print("loading train_loader and val_loader:")
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'train',
                       transform=transforms.Compose([normalize])), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'val',
                       transform=transforms.Compose([normalize])), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    print("Done train_loader and val_loader:")
    # train foreach epoch
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_structure, 0.8)
            adjust_learning_rate(decoder_cell, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
        print("Starting train..............")
        train(train_loader=train_loader,
              encoder=encoder,
              decoder_structure=decoder_structure,
              decoder_cell=decoder_cell,
              criterion_structure=criterion,
              criterion_cell=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_structure_optimizer=decoder_structure_optimizer,
              decoder_cell_optimizer=decoder_cell_optimizer,
              epoch=epoch)
        print("Starting validation..............")
        recent_ted_score = val(val_loader=val_loader, encoder=encoder, decoder_structure=decoder_structure, decoder_cell=decoder_cell, criterion_structure=criterion,
                               criterion_cell=criterion)

        # Check if there was an improvement
        is_best = recent_ted_score > best_TED
        best_TED = max(recent_ted_score, best_TED)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" %
                  (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, encoder, decoder_structure, decoder_cell,
                        encoder_optimizer, decoder_structure_optimizer, decoder_cell_optimizer, recent_ted_score, is_best)


def train(train_loader, encoder, decoder_structure, decoder_cell, criterion_structure, criterion_cell, encoder_optimizer, decoder_structure_optimizer, decoder_cell_optimizer, epoch):

    decoder_structure.train()
    decoder_cell.train()
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    start = time.time()
    print("length of train_loader: {}".format(len(train_loader)))
    for i, (imgs, caption_structures, caplen_structures, caption_cells, caplen_cells, number_cell_per_images) in enumerate(train_loader):
        print("process_batch: {}".format(i))
        imgs = imgs.to(device)
        caption_structures = caption_structures.to(device)
        caplen_structures = caplen_structures.to(device)

        # Foward encoder image and decoder structure
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, hidden_states, sort_ind = decoder_structure(
            imgs, caption_structures, caplen_structures)
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(
            scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(
            targets, decode_lengths, batch_first=True).data
        # Calculate loss
        loss_structures = criterion_structure(scores, targets)

        # Add doubly stochastic attention regularization
        loss_structures += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        # decoder cell per image
        loss_cells = []
        for (i, ind) in enumerate(sort_ind):
            img = imgs[ind]
            hidden_state_structures = hidden_states[i]
            hidden_state_structures = torch.stack(hidden_state_structures)
            number_cell_per_image = number_cell_per_images[ind][0]

            caption_cell = caption_cells[ind][:number_cell_per_image]
            caplen_cell = caplen_cells[ind][:number_cell_per_image]
            caption_cell = caption_cell.to(device)
            caplen_cell = caplen_cell.to(device)

            # Foward encoder image and decoder cell per image
            scores_cell, caps_sorted_cell, decode_lengths_cell, alphas, sort_ind = decoder_cell(
                img, caption_cell, caplen_cell, hidden_state_structures, number_cell_per_image)

            target_cells = caps_sorted_cell[:, 1:]
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_cell = pack_padded_sequence(
                scores_cell, decode_lengths_cell, batch_first=True).data
            target_cells = pack_padded_sequence(
                target_cells, decode_lengths_cell, batch_first=True).data

            loss_cell = criterion_cell(scores_cell, target_cells)
            # Add doubly stochastic attention regularization
            loss_cell += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            loss_cells.append(loss_cell)

            target_cells = caps_sorted_cell[:, 1:]

        # get mean loss_cells
        loss_cells = torch.stack(loss_cells)

        loss_cells = torch.mean(loss_cells)
        loss = hyper_loss * loss_structures + (1-hyper_loss) * loss_cells

        # Back prop.
        decoder_structure_optimizer.zero_grad()
        decoder_cell_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        print("backward..................")
        loss.backward(retain_graph=True)
        loss_structures.backward(retain_graph=True)
        loss_cells.backward()

        if grad_clip is not None:
            clip_gradient(decoder_structure_optimizer, grad_clip)
            clip_gradient(decoder_cell_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_structure_optimizer.step()
        decoder_cell_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
         # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss_structures.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def val(val_loader, encoder, decoder_structure, decoder_cell, criterion_structure, criterion_cell):
    decoder_structure.eval()  # eval mode (no dropout or batchnorm)
    decoder_cell.eval()
    global teds
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    start = time.time()

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    html_trues = list()
    html_predict_only_cells = list()
    html_predict_alls = list()
    with torch.no_grad():
        for i, (imgs, caption_structures, caplen_structures, caption_cells, caplen_cells, number_cell_per_images) in enumerate(val_loader):
            imgs = imgs.to(device)
            caption_structures = caption_structures.to(device)
            caplen_structures = caplen_structures.to(device)

        # Foward encoder image and decoder structure
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, hidden_states, sort_ind_structure = decoder_structure(
                imgs, caption_structures, caplen_structures)

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(
                scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True).data
            # Calculate loss
            loss_structures = criterion_structure(scores, targets)

            # Add doubly stochastic attention regularization
            loss_structures += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            # decoder cell per image
            loss_cells = []
            _, pred_structure = torch.max(scores_copy, dim=2)
            pred_structure = pred_structure.tolist()

            for (i, ind) in enumerate(sort_ind_structure):
                html_predict_only_cell = ""
                html_true = ""
                html_predict_all = ""
                img = imgs[ind]
                hidden_state_structures = hidden_states[i]
                hidden_state_structures = torch.stack(hidden_state_structures)
                number_cell_per_image = number_cell_per_images[ind][0]

                caption_cell = caption_cells[ind][:number_cell_per_image]
                caplen_cell = caplen_cells[ind][:number_cell_per_image]
                caption_cell = caption_cell.to(device)
                caplen_cell = caplen_cell.to(device)

                # Foward encoder image and decoder cell per image
                scores_cell, caps_sorted_cell, decode_lengths_cell, alphas, sort_ind = decoder_cell(
                    img, caption_cell, caplen_cell, hidden_state_structures, number_cell_per_image)
                target_cells = caps_sorted_cell[:, 1:]
                sort_ind = sort_ind.cpu().numpy()

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_cell_copy = scores_cell.clone()
                scores_cell = pack_padded_sequence(
                    scores_cell, decode_lengths_cell, batch_first=True).data
                target_cells = pack_padded_sequence(
                    target_cells, decode_lengths_cell, batch_first=True).data

                loss_cell = criterion_cell(scores_cell, target_cells)
                # Add doubly stochastic attention regularization
                loss_cell += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
                loss_cells.append(loss_cell)

                _, pred_cells = torch.max(scores_cell_copy, dim=2)
                pred_cells = pred_cells.tolist()
                temp_preds = list()
                ground_truth = list()
                # get cell content in per images when predict
                for j, p in enumerate(pred_cells):
                    # because sort cell with descending, mapping pred_cell to sort_ind
                    words = pred_cells[sort_ind[j]
                                       ][:decode_lengths_cell[sort_ind[j]]]
                    temp_preds.append(
                        convertId2wordSentence(id2word_cell, words))

                # get cell content in per images ground_truth

                for j in range(caption_cell.shape[0]):
                    img_caps = caption_cell[j].tolist()
                    img_captions = [w for w in img_caps if w not in {
                        word_map_cell['<start>'], word_map_cell['<pad>']}]  # remove <start> and pads
                    ground_truth.append(convertId2wordSentence(
                        id2word_cell, img_captions))

                index_cell = 0
                cap_structure = caps_sorted[i][:decode_lengths[i]].tolist()
                pred_structure_image = pred_structure[i][:decode_lengths[i]]
                number_cell = 0
                for (index, c) in enumerate(cap_structure):
                    if c == word_map_structure["<start>"] or c == word_map_structure["<end>"]:
                        continue
                    html_predict_only_cell += id2word_stucture[c]
                    html_true += id2word_stucture[c]
                    html_predict_all += id2word_stucture[pred_structure_image[index]]
                    if c == word_map_structure["<td>"] or c == word_map_structure[">"]:
                        html_predict_only_cell += temp_preds[index_cell]
                        html_true += ground_truth[index_cell]
                        html_predict_all += temp_preds[index_cell]
                        index_cell += 1

                print("html_predict: ", html_predict_only_cell)
                print("html_true: ", html_true)
                print("html_predict_all: ", html_predict_all)

                html_predict_only_cells.append(html_predict_only_cell)
                html_predict_alls.append(html_predict_all)
                html_trues.append(html_true)

                # score = teds.evaluate(html_predict_code, html_true_code)
                # print('TEDS score:', score)

                # calculate TEDS for recognition

                # print("number_cell: ", number_cell)

                # get html in per images when predict

            # get mean loss_cells
            loss_cells = torch.stack(loss_cells)

            loss_cells = torch.mean(loss_cells)
            loss = hyper_loss * loss_structures + (1-hyper_loss) * loss_cells

            print("LOSS_STRUCTURE: {} \n LOSS_CELL: {} \n LOSS_DUAL_DECODER: {}".format(
                loss_structures, loss_cells, loss))
            scores_only_cell = teds.batch_evaluate_html(
                html_predict_only_cells, html_trues)

            scores_all = teds.batch_evaluate_html(
                html_predict_alls, html_trues)

            ted_score = np.mean(scores_only_cell)
            print("TED_SCORE: {}".format(ted_score))
            return ted_score

            # temp_preds = list()
            # for j, p in enumerate(preds):
            #     temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            # preds = temp_preds
            # print(preds)
            # hypotheses.extend(preds)


if __name__ == "__main__":
    main()
