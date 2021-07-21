# An automated tool to classify fraudulent credit card transactions using Extreme Gradient Boosting

## Overview

This application was developed by Maximiliano Meyer during his final work titled "Development of a classifying tool to detect false transactions in credit card operations 
in real-time using Machine Learning developed in the Computer Science course at the University of Santa Cruz do Sul - Unisc.<br>
This research and project were developed in the first half of 2021 as a conclusion work of Computer Science course at the University of Santa Cruz do Sul - Unisc.<br>

## Key Topics
The following topics will be covered here:
* Extreme Gradient Boosting
  * Some paramenters used in this tool
* Grid Search CV
* Balanced Bagging Classification

## Extreme Gradient Boosting algorithm

This tool uses the Extreme Gradient Boosting (XGB) algorithm in its construction.<br>
As a boosting algorithm, XGB implements several simpler algorithms in order to achieve a more complete and accurate classification result at the end.<br>
In practice, algorithms of this type work as sequential decision trees since the value that was predicted at <i> n </i> will be taken into account for the prediction at <i> n +1</i> wwhere at every new tree the algorithm will give bigger weight to wrong predictions and smaller for the correct ones. This way with a new set of random columns and values XGB tends to learn how to deal with the peculiarities of the WRONG classification from the previous round.

## Some paramenters used in this tool

One of the highlights of the XGB is precisely the fact that it has dozens of configurable parameters.<br>
These are the main ones used by this tool:

Eta: Represents the learning rate, called eta in the official XGB documentation.<br>
N_estimators: Refers to the number of decision trees that will be created by the model during training.<br>
Min_child_weight: Defines the minimum sum of weights necessary for the tree to continue to be partitioned. The higher this value, the more conservative the algorithm will be and the less superspecific relationships will be learned.<br>
Max_depth: Represents the maximum depth of the tree. Like the previous parameter, it has the same control relationship with overfitting. However, in this case, the higher the more likely it is to overfit.<br>
Subsample: Determines the portion of random training data that will be passed to each tree before they increase by another level.\n'
Colsample_bytree: Similar to the above, determines the portion of columns that will be given randomly when the trees are created.\n'
Scale_pos_weight: Controls the balance between positive and negative weights. It is recommended for cases with a great imbalance between classes.


## Extreme Gradient Boosting algorithm

Balanced Bagging Classification library was also used in this work to deal with unbalanced data as Grid Search CV to make the cross-validation of parameters and values to generate the optimized value



*******************************************************
                        IMPORTANT
                        
Fill in the configuration files in the applications root folder before the first use
for the correct functioning of the application
*******************************************************


Its use is free as long as the source is informed.


For updates, clone or contribute to the project -> https://github.com/Evilmaax<br>
Contact -> MaximilianoMeyer48@gmail.com<br>
Version 0.1
