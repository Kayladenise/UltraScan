import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'train', 'images')
        self.masks_dir = os.path.join(root_dir, 'train', 'masks')

        self.image_filenames = []
        for filename in os.listdir(self.images_dir):
            if filename.endswith('.png'):
                image_id = os.path.splitext(filename)[0]
                mask_filename = f'{image_id}_Annotation.png'
                mask_path = os.path.join(self.masks_dir, mask_filename)
                if os.path.exists(mask_path):
                    self.image_filenames.append(filename)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        image_id = os.path.splitext(img_filename)[0]

        mask_filename = f'{image_id}_Annotation.png'
        mask_path = os.path.join(self.masks_dir, mask_filename)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        print(f"Image Dimensions: {image.shape} - Mask Dimensions: {mask.shape}")


        return image, mask


    

