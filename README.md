# CompetingHiddenUnits
Implementation of "Unsupervised learning by competing hidden units" by Krotov and Hopfield (2019)

Parts are rewritten from https://github.com/DimaKrotov/Biological_Learning/tree/master (c) 2018 Dmitry Krotov -- Apache 2.0 License 


## Run this project

### Option 1: Run Online 
- Open [Google Colab](https://colab.research.google.com/notebook), click on the github tab and paste in the link ```https://github.com/juelha/CompetingHiddenUnits/blob/main/demo.ipynb```
- running the first line of the demo notebook will clone the repo into your google colab files

### Option 2: Run Locally 
- install the conda environment with ```conda env create -f env.yml```
- Clone the Repository or Download the ZIP File and extract.
- Open the demo.ipynb by using your preferred editor program or jupyter 
  
## Performance and Notes

### Current Performance on mnist
<img align="center" width="500" src="https://github.com/juelha/CompetingHiddenUnits/blob/main/reports/mnist/figures/ErrorCurve.png" hspace="10">

This shows the error curve with SGD as the supervised classifier. 
The unsupervised "bio-weight" matrix was learned with 100 hidden units for 200 epochs. 
The dimensions of the weights of the supervised layer were 100x10 for the ten classes. 

The accuracy does not change in any major way despite the decreasing error:

<img align="center" width="500" src="https://github.com/juelha/CompetingHiddenUnits/blob/main/reports/mnist/figures/AccuracyCurve.png" hspace="10">

Note: I found that the prediction of the SGD-classifier tends to trend towards one class per training instance, so I am unsure of how much the classifier actually learns. You can see that by uncommenting the print statements in the accuracy function in Trainer.py.

### Possible reasons for lacking performance
Krotov uses an Adam Optimizer and a decreasing learning rate which is not implemented yet. I am also experimenting with the way of passing the inputs through the unsupervised layer to the classifier since RELU had worse performance.  

Simply increasing the number of hidden units and epochs sadly does not help and seems to worsen the performance:

<img align="center" width="500" src="https://github.com/juelha/CompetingHiddenUnits/blob/main/reports/mnist/figures/ErrorCurve_for2000hidden.png" hspace="10">

(Here the unsupervised layer was trained with 2000 hidden units for 1000 epochs as mentioned in paper.) 

## ToDos
- implement Adam and decreasing learning rate 
- test other supervised classification algorithms than SGD 
- set up experiments for tuning hyperparameters (currently there are taken from the mentioned repo by Krotov)
- implement the slow version of the algorithm ? 
- implement in PyTorch 



## Data Sources:

- mnist in csv format taken from https://pjreddie.com/projects/mnist-in-csv/
- fashion-mnist in csv taken from https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download

