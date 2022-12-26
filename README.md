## A deep generative framework with embedded vector arithmetic and classifier for sample generation, label transfer, and clustering of transcriptome data

This repository contains the official Keras implementation of:

**A deep generative framework with embedded vector arithmetic and classifier for sample generation, label transfer, and clustering of transcriptome data**

**Requirements**
- Python 3.6
- conda 4.4.10
- keras 2.2.4
- tensorflow 1.11.0


**1. Model training**

- *Input*
```
Input1: num_samples * num_genes (--inputdata)
Input2: num_samples * num_covarites (--input_covariates)
Input3: num_samples * num_cellytpe (--inputcelltype)
        supervised: one-hot vector for each sample
        semi-supervised: the num_celltype is set to be larger than the cell types in the reference dataset
                         one-hot vector for sample in the reference dataset
                         zero vector for sample in the query dataset
        unsupervised: the num_celltype is a hyperparameter,  and it is the number of cluster categories set by the user
                      zero vector for each sample
```

- *About this article*
```
#Augments:
#'--inputdata', type=str, default='data/training_data/Fig3_scgen_count7000r.npz', help='address for input data')
#'--input_covariates', type=str, default='data/training_data/Fig3_scgen_study_condition.npy', help='address for covariate (e.g. batch)')
#'--inputcelltype', type=str, default='data/training_data/Fig3_scgen_cell_type.npy', help='address for celltype label')
#'--randoms', type=int, default=30, help='random number to split dataset')
#'--permute_input', type=str, default='T', help='whether permute the input')

#'--dim_latent', type=int, default=50, help='dimension of latent space')
#'--dim_intermediate', type=int, default=200, help='dimension of intermediate layer')
#'--activation', type=str, default='relu', help='activation function: relu or tanh')
#'--arithmetic', type=str, default='minus', help='arithmetic: minus or plus')

#'--batch_size', type=int, default=500, help='training parameters_batch_size')
#'--epochs', type=int, default=50, help='training parameters_epochs')

#'--training', type=str, default='T', help='training model(T) or loading model(F) ')
#'--weights', type=str, default='data/weights_and_results/Fig3_demo.weight', help='trained weights')

#'--mode', type=str, default='supervised', help='mode: supervised, semi_supervised, unsupervised, user_defined')

#'--reconstruction_loss', type=int, default=5, help='The reconstruction loss for VAE')
#'--kl_cross_loss', type=int, default=1, help='')
#'--prior_distribution_loss', type=int, default=1, help='The assumption that prior distribution is uniform distribution')
#'--label_cross_loss', type=int, default=50, help='Loss for integrating label information into the model')


For supervised mode
python inClust.py --inputdata=data/training_data/Fig3_PBMC_count7000r.npz --input_covariates=data/training_data/Fig3_PBMC_study_condition.npy --inputcelltype=data/training_data/Fig3_PBMC_cell_type.npy --mode=supervised --label_cross_loss=20

For semi_supervised mode
python inClust.py --inputdata=data/training_data/Fig5_heart_count.npz --input_covariates=data/training_data/Fig5_heart_batch.npy --inputcelltype=data/training_data/Fig5_heart_label_semi.npy --mode=semi_supervised --permute_input=F

For unsupervised mode
python inClust.py --inputdata=data/training_data/Fig7_count_pca_log.npy --input_covariates=data/training_data/Fig7_img_rgb_smooth50.npy --inputcelltype=data/training_data/Fig7_label.npy --mode=unsupervised  --arithmetic=plus

```

- *Further Explore*
```
Testing your own dataset
python inClust.py --inputdata=your_data --input_covariates=your_inputcelltype --inputcelltype=your_inputcelltype --mode=your_mode
```

- *training weight*
```
results/training.weight
```


**2. Analysis**

- *Demo -- About this article*

The following codes could generate analysis data in the main text.
```
For supervised mode
python inClust.py --inputdata=data/training_data/Fig3_PBMC_count7000r.npz --input_covariates=data/training_data/Fig3_PBMC_study_condition.npy --inputcelltype=data/training_data/Fig3_PBMC_cell_type.npy --mode=supervised --training=F --weights=data/weights_and_results/Fig3_PBMC_demo.weight

For semi_supervised mode
python inClust.py --inputdata=data/training_data/Fig5_heart_count.npz --input_covariates=data/training_data/Fig5_heart_batch.npy --inputcelltype=data/training_data/Fig5_heart_label_semi.npy --mode=semi_supervised --training=F --weights=data/weights_and_results/Fig5_heart_demo.weight

For unsupervised mode
python inClust.py --inputdata=data/training_data/Fig7_count_pca_log.npy --input_covariates=data/training_data/Fig7_img_rgb_smooth50.npy --inputcelltype=data/training_data/Fig7_label.npy --mode=unsupervised  --arithmetic=plus --training=F --weights=data/weights_and_results/Fig7_demo.weight

```

- *Output*

The output is in the results folder, including
- *two latent space representation*
```
mean_vector.npy
Low_dimnesion_vector.npy
batch_vector.npy
```

- *predictd label for each sample*
```
predict_labels.csv
```
**3. reproduction figures**
```
demo_reproducing_figures.ipynb
```

**VAE and its variant implementation**
https://github.com/bojone/vae