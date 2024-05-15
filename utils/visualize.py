import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from utils.data_loader import CustomDataset

def visualize_image_mask(image, mask, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def main():
    data_dir = r'C:\Users\kayla\OneDrive\Desktop\UltraScan Fetal Head Segmentation\data\train'
    image_filename = '1_HC.png'
    annotation_filename = '1_HC_Annotation.png'
    
    image_path = os.path.join(data_dir, 'images', image_filename)
    annotation_path = os.path.join(data_dir, 'images', annotation_filename)
    
    image = Image.open(image_path).convert('RGB')
    annotation = Image.open(annotation_path).convert('L')  
    
    visualize_image_mask(image, annotation)
    
    save_path = 'visualization_example.png'
    visualize_image_mask(image, annotation, save_path=save_path)
    print(f"Visualization saved at '{save_path}'.")

if __name__ == "__main__":
    main()
