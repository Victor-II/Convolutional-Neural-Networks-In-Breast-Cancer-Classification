# Convolutional-Neural-Networks-In-Breast-Cancer-Classification



### Models
- ResNet-50 (pretrained & not pretrained)
- DenseNet-121 (pretrained & not pretrained)
- InceptionNet-V3 (pretrained & not pretrained)

### Breakhis Dataset
The Breast Cancer Histopathological Image Classification (BreakHis) is composed of 7,909 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X). To date, it contains 2,480 benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format) [^1].

### Training Procedure  
- DEVICE > cuda
- BATCH SIZE > 16
- EPOCHS > 100
- CRITERION > Binary Cross-Entropy Loss
- OPTIMIZER > ADAM (LR = 0.001 > not pretrained / LR = 0.0001 > pretrained)
- LR SCHEDULER > ReduceLROnPlateau (factor=0.25, patience=5, threshold=0.001)

[^1]: https://www.kaggle.com/datasets/ambarish/breakhis 
