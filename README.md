# opt_ml_project

## Step 1

Definition of the notion of generalization (test error, image augmentation, initialization)
(Question: Will the hyper parameter changes drastictly between different image detection tasks)

## Step 2

Model and Datasets selection. Model is generic CNN model (ResNet). Datasets are standard computer vision datasets: MNIST, FASHION MNIST, CIFAR10, CIFAR100

## Step 3

Dataset augmentation. To test for generalization we will augment our dataset by adding for each image in it:
- Salt and Pepper noise (1x)
- Random affine transform (2x)
- 

## Comparison Framework
- Start from same set of starting points
- Choose best set of hyper parameters 
- Run until convergence
- Comparison points will be final accuracy, time to convergence, accuracy on transformed dataset for each starting points.

