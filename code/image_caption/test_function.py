import torch
import torchvision.transforms as transforms
from dataset import *

batch_size = 2
data_folder = "output"
workers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, 'TRAIN',
                   transform=transforms.Compose([normalize])),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

for i, (imgs, caption_structures, caplen_structures, caption_cells, caplen_cells, number_cell_per_images) in enumerate(train_loader):
    imgs = imgs.to(device)
    caption_structures = caption_structures.to(device)
    caplen_structures = caplen_structures.to(device)

    caption_cells = caption_cells.to(device)
    caplen_cells = caplen_cells.to(device)
    number_cell_per_images = number_cell_per_images.to(device)

    print("caption structure size: ", caption_structures.size())
    print("caplen structure size: ", caplen_structures.size())
    print("caption cell size: ", caption_cells.size())
    print("caplen cell size: ", caplen_cells.size())
    print("number_cell_per_images: ", number_cell_per_images.size())
    continue
