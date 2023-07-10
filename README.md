# Convolutional-Neural-Networks-In-Breast-Cancer-Classification

## Summary

The study aimed to compare the performance of 3 separate CNN models, both pretrained and trained from scratch, in the task of histopathological image classification. All models have beeen trained according to the procedure presented below. Moreover, each network's classifier has been adapted to the task of binary classification, by reducing the number of units of the last fully-connected layer to 1. It was found that pretrained models produce superior results in terms of accuracy while also being more time efficient, having reached peak performance almost twice as fast as their trained from scratch counterparts.  


## Models
- ResNet-50 (pretrained & not pretrained)
- DenseNet-121 (pretrained & not pretrained)
- InceptionNet-V3 (pretrained & not pretrained)

## Breakhis Dataset
The Breast Cancer Histopathological Image Classification (BreakHis) is composed of 7,909 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X). To date, it contains 2,480 benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format) [^1].

## Training Parameters  
- DEVICE: Cuda
- BATCH SIZE: 16
- EPOCHS: 100
- CRITERION: Binary Cross-Entropy Loss
- OPTIMIZER: ADAM (LR = 0.001 > not pretrained / LR = 0.0001 > pretrained)
- LR SCHEDULER: ReduceLROnPlateau (factor=0.25, patience=5, threshold=0.001)

## Results

### Not Pretrained

#### DenseNet-121  

| | Precision | Recall | F1 Score | Accuracy |
|--|-----------|---------|---------|----------|
| Benign | 0.9315 | 0.9315 | 0.9315 | 0.9571 |
| Malign | 0.9688 | 0.9688 | 0.9688 | 0.9571 |
| Mean | 0.9501 | 0.9501 | 0.9501 | 0.9571 |
| Weighted mean | 0.9571 | 0.9571 | 0.9571 | 0.9571 |

#### InceptionNetV3

| | Precision | Recall | F1 Score | Accuracy |
|--|-----------|---------|---------|----------|
| Benign | 0.9352 | 0.9315 | 0.9333 | 0.9583 |
| Malign | 0.9688 | 0.9706 | 0.9697 | 0.9583 |
| Mean | 0.9520 | 0.9510 | 0.9515 | 0.9583 |
| Weighted mean | 0.9583 | 0.9583 | 0.9583 | 0.9583 |

#### ResNet-50 

| | Precision | Recall | F1 Score | Accuracy |
|--|-----------|---------|---------|----------|
| Benign | 0.9246 | 0.9395  | 0.9320 | 0.9571 |
| Malign | 0.9722 | 0.9651 | 0.9686  | 0.9571 |
| Mean | 0.9484 | 0.9523 | 0.9503 | 0.9571 |
| Weighted mean | 0.9573 | 0.9571 | 0.9572 | 0.9571 |

### Pretrained
---
#### DenseNet-121

| | Precision | Recall | F1 Score | Accuracy |
|--|-----------|---------|---------|----------|
| Benign | 0.9759 | 0.9798 | 0.9779 | 0.9861 |
| Malign | 0.9908 | 0.9890  | 0.9899 | 0.9861 |
| Mean | 0.9833 | 0.9844 | 0.9839 | 0.9861 |
| Weighted mean | 0.9861 | 0.9861 | 0.9861 | 0.9861 |

#### InceptionNet-V3

| | Precision | Recall | F1 Score | Accuracy |
|--|-----------|---------|---------|----------|
| Benign | 0.9839 | 0.9839 | 0.9839 | 0.9899 |
| Malign | 0.9926 | 0.9926  | 0.9926 | 0.9899 |
| Mean | 0.9883 | 0.9883 | 0.9883 | 0.9899 |
| Weighted mean | 0.9899 | 0.9899 | 0.9899 | 0.9899 |
  
#### ResNet-50  

| | Precision | Recall | F1 Score | Accuracy |
|--|-----------|---------|---------|----------|
| Benign | 0.9759 | 0.9798 | 0.9779 | 0.9861 |
| Malign | 0.9908 | 0.9890 | 0.9899 | 0.9861 |
| Mean | 0.9833 | 0.9844 | 0.9839 | 0.9861 |
| Weighted mean | 0.9861 | 0.9861 | 0.9861 | 0.9861 |

[^1]: https://www.kaggle.com/datasets/ambarish/breakhis 
