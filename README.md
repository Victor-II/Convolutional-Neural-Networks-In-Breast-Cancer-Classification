# Convolutional-Neural-Networks-In-Breast-Cancer-Classification



### Models
- ResNet-50 (pretrained & not pretrained)
- DenseNet-121 (pretrained & not pretrained)
- InceptionNet-V3 (pretrained & not pretrained)

### Breakhis Dataset
The Breast Cancer Histopathological Image Classification (BreakHis) is composed of 7,909 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X). To date, it contains 2,480 benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format) [^1].

### Training Parameters  
- DEVICE: Cuda
- BATCH SIZE: 16
- EPOCHS: 100
- CRITERION: Binary Cross-Entropy Loss
- OPTIMIZER: ADAM (LR = 0.001 > not pretrained / LR = 0.0001 > pretrained)
- LR SCHEDULER: ReduceLROnPlateau (factor=0.25, patience=5, threshold=0.001)

### Results
#### Pretrained
DenseNet-121  

 | | Precision | Recall | F1 Score | Accuracy |
|--|-----------|---------|---------|----------|
| Benign | 0.9315 | 0.9315 | 0.9315 | |
| Malign | 0.9688 | 0.9688 | 0.9688 | 0.9571 |
| Medie | 0.9501 | 0.9501 | 0.9501 | |
| Medie Ponderată | 0.9571 | 0.9571 | 0.9571 | |


InceptionNetV3
Benign 0.9352 0.9315 0.9333
Malign 0.9688 0.9706 0.9697 0.9583
Medie 0.9520 0.9510 0.9515
Medie Ponderată 0.9583 0.9583 0.9583
ResNet-50
Benign 0.9246 0.9395 0.9320
Malign 0.9722 0.9651 0.9686 0.9571
Medie 0.9484 0.9523 0.9503
Medie Ponderată 0.9573 0.9571 0.9572
[^1]: https://www.kaggle.com/datasets/ambarish/breakhis 
