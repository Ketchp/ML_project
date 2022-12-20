# Introduction to Machine learning and Data mining

This repository contains source code for group project for
class Introduction to Machine learning and Data mining.

Project uses 'adult' dataset for all tasks.
Project focuses on data preparation/exploration, classification and regression.

[Report 1](report_1.pdf) is first introduction to dataset.
It mainly contains figures describing dataset and PCA decomposition.
Data attributes of original 'Adult' dataset are also transformed to deal with issues like class imbalance, or nominal attributes.

[Report 2](report_2.pdf) contains results of regression and classification.
Both regression and classification uses two level cross-validation to estimate ideal model parameters and generalisation error.
Regression part tries to predict persons age from other available attributes.
We compared regularised linear regressor and artificial neural network models for this task.
Classification part predicts persons earning class from other attributes.
We used regularised logistic regression and KNN classifier for this task.
Each part of report quicly discusses results of training and what we can extract about data from trained model.