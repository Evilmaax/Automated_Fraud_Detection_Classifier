# An automated tool to classify fraudulent credit card transactions using Extreme Gradient Boosting

## Overview

This application was developed by Maximiliano Meyer during his final work titled "Development of a classifying tool to detect false transactions in credit card operations 
in real-time using Machine Learning developed in the Computer Science course at the University of Santa Cruz do Sul - Unisc.

This research and project were developed in the first half of 2021 as a conclusion work of Computer Science course at the University of Santa Cruz do Sul - Unisc.<br>

## Key Topics
The following topics will be covered here:
* Extreme Gradient Boosting
  * Some paramenters used in this tool
* Grid Search CV
* Balanced Bagging Classifier

### Extreme Gradient Boosting algorithm

This tool uses the Extreme Gradient Boosting (XGB) algorithm in its construction.

As a boosting algorithm, XGB implements several simpler algorithms in order to achieve a more complete and accurate classification result at the end.<br>
In practice, algorithms of this type work as sequential decision trees since the value that was predicted at <i> n </i> will be taken into account for the prediction at <i> n +1</i> where at every new tree the algorithm will give bigger weight to wrong predictions and smaller for the correct ones. 

This way with a new set of random columns and values XGB tends to learn how to deal with the peculiarities of the WRONG classification from the previous round.

### Some paramenters used in this tool

One of the highlights of the XGB is precisely the fact that it has dozens of configurable parameters.<br>
These are the main ones used by this tool:

* <i>Eta</i>: Represents the learning rate, called eta in the official XGB documentation.<br>
* <i>N_estimators</i>: Refers to the number of decision trees that will be created by the model during training.<br>
* <i>Min_child_weight</i>: Defines the minimum sum of weights necessary for the tree to continue to be partitioned. The higher this value, the more conservative the algorithm will be and the less superspecific relationships will be learned.<br>
* <i>Max_depth</i>: Represents the maximum depth of the tree. Like the previous parameter, it has the same control relationship with overfitting. However, in this case, the higher the more likely it is to overfit.<br>
* <i>Subsample</i>: Determines the portion of random training data that will be passed to each tree before they increase by another level.'
* <i>Colsample_bytree</i>: Similar to the above, determines the portion of columns that will be given randomly when the trees are created.'
* <i>Scale_pos_weight</i>: Controls the balance between positive and negative weights. It is recommended for cases with a great imbalance between classes.

### Grid Search CV

Grid Search CV is a module of the Scikit Learn library used to automate the parameter optimization process that XGB makes necessary. In addition to creating as many simulations as necessary through a data crossing, the tool is also capable of evaluating the performance of each of these arrangements using a metric defined by the user.

To use it, we just need to define which parameters will be tested, which will be the possible values for each scenario and the number of Cross Validations (CV) to be performed. The CV value represents the number of divisions in which the dataset will be partitioned. The technique consists of generating <i>n<\i> cuts of the same size in order to achieve a greater variety of training and testing scenarios in a simulation. 
 
For example: If we use a dataset with 70 thousand records and define CV value being 7, then this dataset will be divided into 7 parts of 10000 records each. Subsequently, these 7 datasets will undergo training and validation, alternating which part is being used as test and training until all combinations are performed, as shown below.

![Example of 7 parts Cross-Validation](https://i.stack.imgur.com/padg4.gif)

[gif source](https://stackoverflow.com/questions/31947183/how-to-implement-walk-forward-testing-in-sklearn)
 
http://github.com - automatic!
[GitHub](http://github.com)
 
## Balanced Bagging Classifier

Part of the <i>Imbalanced Learn</i> library (a branch of Scikit Learn) this tool aims to diminish the damage of highly unbalanced classes. Using it, it is possible to create records for classes with undersampling, remove records in classes with oversampling and perform resampling, a technique in which the small amount of data available from an unbalanced class is used to estimate a population parameter.



*******************************************************
                        IMPORTANT
                        
Fill in the configuration files in the applications root folder before the first use
for the correct functioning of the application
*******************************************************


Its use is free as long as the source is informed.


For updates, clone or contribute to the project -> https://github.com/Evilmaax<br>
Contact -> MaximilianoMeyer48@gmail.com<br>
Version 0.1
