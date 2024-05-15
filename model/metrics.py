import torch
import numpy as np
from sklearn.metrics import confusion_matrix

class ConfusionMatrix:
    def __init__(self, nclasses, classes, useUnlabeled=False):
        self.mat = np.zeros((nclasses, nclasses), dtype=float)
        self.valids = np.zeros((nclasses), dtype=float)
        self.IoU = np.zeros((nclasses), dtype=float)
        self.mIoU = 0

        self.nclasses = nclasses
        self.classes = classes
        self.list_classes = list(range(nclasses))
        self.useUnlabeled = useUnlabeled
        self.matStartIdx = 1 if not self.useUnlabeled else 0

    def update_matrix(self, target, prediction):
        temp_target = None
        temp_prediction = None

        if not isinstance(prediction, np.ndarray) or not isinstance(target, np.ndarray):
            print("Expecting ndarray")
            return
        elif len(target.shape) == 3:  
            if len(prediction.shape) == 4: 
                temp_prediction = np.argmax(prediction, axis=1).flatten()
            elif len(prediction.shape) == 3:  
                temp_prediction = prediction.flatten()
            else:
                print("Make sure prediction and target dimension is correct (batch, height, width) for target and (batch, classes, height, width) or (batch, height, width) for prediction")
                return

            temp_target = target.flatten()
        elif len(target.shape) == 2:  
            if len(prediction.shape) == 3:  
                temp_prediction = np.argmax(prediction, axis=0).flatten()
            elif len(prediction.shape) == 2: 
                temp_prediction = prediction.flatten()
            else:
                print("Make sure prediction and target dimension is correct (height, width) for both")
                return

            temp_target = target.flatten()
        elif len(target.shape) == 1:  
            if len(prediction.shape) == 2:  
                temp_prediction = np.argmax(prediction, axis=0).flatten()
            elif len(prediction.shape) == 1:  
                temp_prediction = prediction
            else:
                print("Make sure prediction and target dimension is correct (n_samples) for both")
                return

            temp_target = target
        else:
            print("Data with this dimension cannot be handled")
            return

        # update confusion matrix
        if temp_target is not None and temp_prediction is not None:
            self.mat += confusion_matrix(temp_target, temp_prediction, labels=self.list_classes)

    def scores(self):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        total = 0   # Total true positives
        N = 0       # Total samples
        
        for i in range(self.matStartIdx, self.nclasses):
            N += sum(self.mat[:, i])
            tp = self.mat[i][i]
            fp = sum(self.mat[self.matStartIdx:, i]) - tp
            fn = sum(self.mat[i, self.matStartIdx:]) - tp

            if (tp + fp) == 0:
                self.valids[i] = 0
            else:
                self.valids[i] = tp / (tp + fp)

            if (tp + fp + fn) == 0:
                self.IoU[i] = 0
            else:
                self.IoU[i] = tp / (tp + fp + fn)

            total += tp

        self.mIoU = sum(self.IoU[self.matStartIdx:]) / (self.nclasses - self.matStartIdx)
        self.accuracy = total / (sum(sum(self.mat[self.matStartIdx:, self.matStartIdx:])))

        return self.valids, self.accuracy, self.IoU, self.mIoU, self.mat

    def plot_confusion_matrix(self, filename):
        print(f"Plotting confusion matrix to {filename}...")

    def reset(self):
        self.mat = np.zeros((self.nclasses, self.nclasses), dtype=float)
        self.valids = np.zeros((self.nclasses), dtype=float)
        self.IoU = np.zeros((self.nclasses), dtype=float)
        self.mIoU = 0

def dice_coefficient(predicted, target, smooth=1e-6):
    # calculate dice coefficient 4 binary segmentation
    predicted = predicted.view(-1)
    target = target.view(-1)
    
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item() 

def intersection_over_union(predicted, target, smooth=1e-6):
    # calculate (IoU) 4 binary segmentation
    predicted = predicted.view(-1)
    target = target.view(-1)
    
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def accuracy(predicted, target):
    # calculate pixel accuracy
    correct = (predicted == target).sum().item()
    total_pixels = target.numel()
    
    acc = correct / total_pixels
    return acc
