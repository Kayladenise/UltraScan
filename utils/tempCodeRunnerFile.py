import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from model.linknet import MultiScaleLinkNet
from utils.data_loader import CustomDataset
from model.metrics import dice_coefficient, intersection_over_union, accuracy

from utils.visualize import visualize_image_mask
from utils.confusion_matrix import ConfusionMatrix

def main():
    print("Starting Fetal Head Segmentation...")
    data_dir = 'C:\\Users\\kayla\\OneDrive\\Desktop\\UltraScan Fetal Head Segmentation\\data'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(root_dir=data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = MultiScaleLinkNet(in_channels=3, num_classes=1)
    criterion = nn.BCEWithLogitsLoss() #Binary cross entropy(for binary classification)
    optimizer = optim.Adam(model.parameters(), lr=0.001) #using Adam optimizer

    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    confusion_matrix = ConfusionMatrix(nclasses=2, classes=['background', 'fetal_head'])

    best_dice = 0.0
    best_model_path = 'fetal_head_segmentation_model.pth'

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        dice_total = 0.0
        iou_total = 0.0
        acc_total = 0.0
        num_batches = 0

        print(f'Epoch [{epoch+1}/{num_epochs}], Training...')

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            #optimization steps
            optimizer.zero_grad() #to clear previously accumulated gradients
            outputs = model(images) #
            loss = criterion(outputs, masks)
            loss.backward() #backpropogate the loss to compute gradients
            optimizer.step() #update model parameters

            epoch_loss += loss.item()

            predicted = torch.sigmoid(outputs) > 0.5
            target = (masks > 0.5).float()
            dice_total += dice_coefficient(predicted, target)
            iou_total += intersection_over_union(predicted, target)
            acc_total += accuracy(predicted, target)
            num_batches += 1
            
            confusion_matrix.update_matrix(target.cpu().numpy(), predicted.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                avg_dice = dice_total / num_batches
                avg_iou = iou_total / num_batches
                avg_acc = acc_total / num_batches
                print(f'  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, '
                      f'Avg Dice: {avg_dice:.4f}, Avg IoU: {avg_iou:.4f}', 'Avg Acc: {avg_acc:.4f}')

        avg_loss = epoch_loss / len(train_loader)
        avg_dice = dice_total / num_batches
        avg_iou = iou_total / num_batches
        avg_acc = acc_total / num_batches

        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, '
              f'Avg Dice: {avg_dice:.4f}, Avg IoU: {avg_iou:.4f}', 'Avg Acc: {avg_acc:.4f}')


        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved model with best Dice coefficient: {best_dice:.4f}')
            
        valids, accuracy, IoU, mIoU, confusion_mat = confusion_matrix.scores()        
        print(f'  Validations: {valids}')
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  Mean IoU: {mIoU:.4f}')
        print(f'  Confusion Matrix:\n{confusion_mat}')

        example_image, example_mask = next(iter(train_loader))
        example_image = example_image.to(device)
        example_output = model(example_image)
        example_predicted = torch.sigmoid(example_output) > 0.5

        image_np = example_image[0].cpu().numpy().transpose((1, 2, 0))
        mask_np = example_mask[0].numpy().squeeze()
        predicted_np = example_predicted[0].cpu().numpy().squeeze()

        visualize_image_mask(image_np, mask_np)
        save_path = f'visualization_epoch_{epoch + 1}.png'
        visualize_image_mask(image_np, predicted_np, save_path=save_path)
        print(f"Visualization saved at '{save_path}'.")

    print(f"Training completed. Best model saved at '{best_model_path}'.")

if __name__ == "__main__":
    main()
