# Deep Learning Reproducibility Project: A Novel Hybrid Convolutional Neural Network for Accurate Organ Segmentation in 3D Head and Neck CT Images

This repository is an attempt at reproducing the results from, 
*A Novel Hybrid Convolutional Neural Network for Accurate Organ Segmentation
in 3D Head and Neck CT Images[1]*. This repository is implementation  in the pytorch framework.
For an elaborate overview of our reproduction and results please see our blog post at, 
https://thomas100z.github.io/DLgroup79/


## Installation
Install required packages:
```
python3 -m pip install -r requirements.txt
```
## Getting started

### Data
Extract the [data](https://github.com/prerakmody/hansegmentation-uncertainty-qa/releases) to the `data` folder.
The data folder should be structured like this:
```
.
├── ...
├── data                    
│   ├── test_offsite                # test set
│   │   └── data_3D
│   │       └── ...                 # samples
│   ├── train                       # train set 
│   │    └── data_3D
│   │        └── ...                # samples
│   └── train_additional            # validation set  
│        └── data_3D
│            └── ...                # samples 
└── ...
```

For linux this snippet can be run:
```
wget https://github.com/prerakmody/hansegmentation-uncertainty-qa/releases/download/v1.0/train_additional.zip
wget https://github.com/prerakmody/hansegmentation-uncertainty-qa/releases/download/v1.0/train.zip
wget https://github.com/prerakmody/hansegmentation-uncertainty-qa/releases/download/v1.0/test_offsite.zip

unzip train.zip -d data/
unzip train_additional.zip -d data/
unzip test_offsite.zip -d data/

rm train.zip
rm train_additional.zip
rm test_offsite.zip
```
### Training model
To train the model simply run the following command:
```
python3 main.py
```

### Testing model
To test the model on the test set and view the box plot and inferencing results run:

```
python3 predict.py  # (optionally provide model to test as firts argument, default last model is used): ./models/xxx.pth
```

## Results
Background,training & testing results and discussion can be found in our [blog post](https://thomas100z.github.io/DLgroup79/).

## Acknowledgements
We would like to thank [Prerak Mody](https://github.com/prerakmody) for his help and insight during the project as well as for his pre-procssed data from the MICCAI2015 challenge [2].

## Contributors
Thomas Zuiker  
Storm Holman  
Lucas Veeger   
Emma Botten  

## References 
**Original paper**  
A Novel Hybrid Convolutional Neural Network for Accurate Organ Segmentation in 3D Head and Neck CT Images  
By: Zijie Chen, Cheng Li, Junjun He, Jin Ye, Diping Song, Shanshan Wang, Lixu Gu, and Yu Qiao  
DOI: https://doi.org/10.1007/978-3-030-87193-2_54


**Dataset**  
2015 MICCAI Challenge: Evaluation of segmentation methods on head and neck CT: Auto‐segmentation challenge 2015.  
By: Raudaschl, P. F., Zaffino, P., Sharp, G. C., Spadea, M. F., Chen, A., Dawant, B. M., ... & Jung, F.  
Obtained from: https://github.com/prerakmody/hansegmentation-uncertainty-qa/releases
