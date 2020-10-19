import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataset import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

data_folder = "output"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
checkpoint = ""
word_map_structure_file = "output/WORDMAP_STRUCTURE.json"
word_map_cell_file = "output/WORDMAP_CELL.json"
# Load model
checkpoint = torch.load(checkpoint)
decoder_structure = checkpoint['decoder_structure']
decoder_cell = checkpoint["decoder_cell"]
decoder_structure = decoder_structure.to(device)
decoder_cell.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_structure_file, 'r') as j:
    word_map_structure = json.load(j)
with open(word_map_cell_file, "r") as j:
    word_map_cell = json.load(j)

id2word_stucture = id_to_word(word_map_structure)
id2word_cell = id_to_word(word_map_cell)

vocab_size_structure = len(word_map_structure)
vocab_size_cell = len(word_map_cell)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def evaluation()