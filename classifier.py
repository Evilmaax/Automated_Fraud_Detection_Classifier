import os, random, sys, time
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV as CV
from sklearn.metrics import plot_confusion_matrix, recall_score, matthews_corrcoef

root_folder = os.path.dirname(sys.argv[0])

try:
    file = open(root_folder + "/config.txt", "r")
    config = file.readlines()

    label = config[0].split(" #")[0]
    to_keep = config[1].split(" #")[0].split(',')
    correlationRate = float(config[2].split(" #")[0])
    eta = float(config[3].split(" #")[0])
    n_estimators = int(config[4].split(" #")[0])
    min_child_weight = int(config[5].split(" #")[0])
    subsample = float(config[6].split(" #")[0])
    colsample_bytree = float(config[7].split(" #")[0])
    scale_pos_weight = int(config[8].split(" #")[0])
    max_depth = int(config[9].split(" #")[0])
    split = float(config[10].split(" #")[0])

    file.close()

except:
    print("\nConfig file cound not be loaded.\nPlease verify")
    sys.exit()


def loadFile():
    try:
        files = os.listdir(f"{root_folder}/Dataset")

        print("\nWhich file do you want to use in this operation?\n")
        for i in range(len(files)):
            print(f"{i + 1} - {files[i]}")
        op = int(input("\nAnswer: "))

        print("\nLoading data.\nThis can take a while based on the size of the file")
        data = pd.read_csv(root_folder + '/Dataset/' + files[op - 1])
        print("\n" + files[op - 1] + " successfully loaded!")

        return data

    except:
        print("\nCould not load the file\nPlease verify if the folder 'Dataset' has something inside"
              " and it is a file with .CSV extension")
        sys.exit()


def optimize():
    data = loadFile()

    cv = int(input("\nHow many folds do you want to use?"
                   "\nBetter solutions starts at 10 folds: "))

    round, totalTime = 1, time.time()

    parameters = [['eta', 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                  ['n_estimators', 300, 400, 500, 600, 700, 800],
                  ['min_child_weight', 3, 4, 5, 6, 7],
                  ['subsample', 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  ['colsample_bytree', 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  ['scale_pos_weight', 3, 4, 5, 6, 7],
                  ['max_depth', 5, 6, 7, 8, 9, 10],
                  ['split', 0.2, 0.25, 0.3, 0.35]]

    available = []

    for x in range(len(parameters)):
        if len(parameters[x]) > 2:
            available.append(x)

    while len(available) > 0:
        print("\nRound ", round)
        start = time.time()
        paramName, paramValues = [], []

        if len(available) > 3:
            temp = random.sample(available, 3)
        else:
            temp = random.sample(available, len(available))

        for x in temp:
            paramName.append(parameters[x][0])
        for x in range(len(temp)):
            tempList = []
            if len(parameters[temp[x]]) > 3:
                temp2 = random.sample(range(1, len(parameters[temp[x]])), 3)
            else:
                temp2 = random.sample(range(1, len(parameters[temp[x]])), len(parameters[temp[x]]) - 1)

            for y in range(len(temp2)):
                tempList.append(parameters[temp[x]][temp2[y]])
            paramValues.append(tempList)

        toBeTested = dict(zip(paramName, paramValues))

        print("    Tested values: ", toBeTested)
        grid = CV(estimator=XGBClassifier(verbosity=0, use_label_encoder=False),
                  param_grid=toBeTested,
                  scoring='recall',
                  n_jobs=-1,
                  cv=cv)

        grid.fit(data.drop([label], axis='columns'), data[label])
        bestScenario = grid.best_params_
        print("    The best combination for tested values is: ", bestScenario)

        for key, value in bestScenario.items():
            for x in range(len(parameters)):
                if parameters[x][0] == key:
                    y = 0
                    while y < len(parameters[x]):
                        if (parameters[x][y] != value) and (parameters[x][y] in toBeTested[key]):
                            parameters[x].pop(y)
                            y = 0
                        y += 1

        round += 1
        print(f"    Elapsed time on this round: {time.time() - start:.3f} seconds")

        available = []

        for x in range(len(parameters)):
            if len(parameters[x]) > 2:
                available.append(x)

    print(f"\nTotal time: {time.time() - totalTime:.3f} seconds")

    with open("optimized.txt", "w") as newConfig:
        for line in parameters:
            line = " ".join(map(str, line))
            line1, line2 = line.split(' ')
            line2 = line2.strip() + " # " + line1 + "\n"
            newConfig.write(line2)

    file = open(root_folder + "/optimized.txt", "r")
    optimizedConfig = file.readlines()
    print("File optimized.txt exported successfully to the root folder")


    eta = float(optimizedConfig[0].split(" #")[0])
    n_estimators = int(optimizedConfig[1].split(" #")[0])
    min_child_weight = int(optimizedConfig[2].split(" #")[0])
    subsample = float(optimizedConfig[3].split(" #")[0])
    colsample_bytree = float(optimizedConfig[4].split(" #")[0])
    scale_pos_weight = int(optimizedConfig[5].split(" #")[0])
    max_depth = int(optimizedConfig[6].split(" #")[0])
    split = float(optimizedConfig[7].split(" #")[0])

    print("\nOptimization finished. Selected values:\n\n"
          f"Eta: {eta}\n"
          f"n_estimators: {n_estimators}\n"
          f"min_child_weight: {min_child_weight}\n"
          f"subsample: {subsample}\n"
          f"colsample_bytree: {colsample_bytree}\n"
          f"scale_pos_weight: {scale_pos_weight}\n"
          f"max_depth: {max_depth}\n"
          f"taxa de split: {split}")

    file.close()

    input("\nPress enter to continue")
    menu()


def standardize(data):

    file = open(root_folder + "/columns.txt", "r")
    columns = file.read().splitlines()
    file.close()

    for i in data.columns:
        if i not in columns:
            data.drop([i], axis='columns', inplace=True)
        else:
            data[i].fillna(0, inplace=True)

    return data


def classify(model):
    totalTime = time.time()
    data = loadFile()

    print("\nClassification process started...")
    data = standardize(data)

    data[label] = model.predict(data)
    data.to_csv(root_folder + '/Dataset/classification_result.csv', index=False)

    print('\nFile classification_result.csv saved sucessffully to Dataset folder')
    print(f"Classification total time: {time.time() - totalTime:.3f} seconds")


def training():
    totalTime = time.time()
    data = loadFile()

    trueTransactions = data[label].value_counts()[0]
    print(trueTransactions)
    fakeTransactions = data[label].value_counts()[1]
    totalTransactions = data[label].count()

    print(f'\nSome infos about your dataset:\n\nTotal transactions: {totalTransactions}\nTrue transactions: {trueTransactions}\nFalse transactions: {fakeTransactions}\nPercentage of false transactions: {(fakeTransactions / totalTransactions) * 100:.3f}%')

    bbc = BalancedBaggingClassifier(base_estimator=XGBClassifier(
        eta=eta,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        max_depth=max_depth,
        objective='binary:logistic',
        use_label_encoder=False,
        verbosity=0),
        sampling_strategy='all',
        n_jobs=-1)

    X_train, X_test, y_train, y_test = train_test_split(data.drop([label], axis='columns'),
                                                        data[label], test_size=split, stratify=data[label])

    bbc.fit(X_train, y_train)
    joblib.dump(bbc, root_folder + '/model.pkl')
    print('\nModel created and saved successfully on the root folder')

    y_pred = bbc.predict(X_test)

    print("\nYour training values:")
    print("Recall: ", recall_score(y_test, y_pred))

    plot_confusion_matrix(bbc, X_test, y_test, values_format='d',
                          display_labels=['True transactions', 'False transactions'])

    print(f"\nTotal time: {time.time() - totalTime:.3f} seconds")

    plt.show()
    input("\nPress enter to continue")
    menu()


def preProcessing():
    data = loadFile()
    print("\nPreprocessing started...")

    totalTime = time.time()
    correlatedColumns = set()
    correlatedMatrix = data.corr()

    for i in range(len(correlatedMatrix.columns)):
        for j in range(i):
            if abs(correlatedMatrix.iloc[i, j]) > correlationRate:
                colname = correlatedMatrix.columns[i]
                if colname not in to_keep:
                    correlatedColumns.add(colname)

    data.drop(correlatedColumns, axis=1, inplace=True)

    print("\nCorrelated columns removed")
    nanValues = set()

    for y in data.columns:
        if data[y].dtype == object:
            nanValues.add(y)

    data.drop(nanValues, axis=1, inplace=True)

    print("Non numeric columns removed")

    for i in data.columns:
        data[i].fillna(0, inplace=True)

    print("Not a number values filled")

    with open("columns.txt", "w+") as columns:
        for i in data.columns:
            if i != label:
                columns.write(i)
                columns.write("\n")

    data.to_csv(root_folder + '/Dataset/preprocessing_done.csv', index=False)
    print("\nPreprocessing finished")
    print("File preprocessing_done.csv created and saved successfully on the Dataset folder")
    print(f"\nTotal time: {time.time() - totalTime:.3f} seconds")

    input("\nPress enter to continue")
    menu()


def menu():
    options, op = [1, 2, 3, 4, 5, 6, 7], 0

    while op not in options:
        op = int(input("\nSelect an option below:\n\n" 
                       "1 - Apply preprocessing to the dataset (Choose this option if this is your first time using it)\n"
                       "2 - Train the model using the default configuration\n"
                       "3 - Train the model Using the optimized configuration\n"
                       "4 - Test combinations of parameters and values to generate an optimized configuration\n"
                       "5 - Classify transactions using the previously trained model\n"
                       "6 - Help\n"
                       "7 - About\n\n"
                       "Answer: "))

    if op == 1:
        preProcessing()

    elif op == 2:
        try:
            file = open(root_folder + "/config.txt", "r")
            config = file.readlines()

            eta = float(config[3].split(" #")[0])
            n_estimators = int(config[4].split(" #")[0])
            min_child_weight = int(config[5].split(" #")[0])
            subsample = float(config[6].split(" #")[0])
            colsample_bytree = float(config[7].split(" #")[0])
            scale_pos_weight = int(config[8].split(" #")[0])
            max_depth = int(config[9].split(" #")[0])
            split = float(config[10].split(" #")[0])

            file.close()

            print(f"\nSuccessfully loaded default values:\n\n"
                  f"Eta: {eta}\n"
                  f"n_estimators: {n_estimators}\n"
                  f"min_child_weight: {min_child_weight}\n"
                  f"subsample: {subsample}\n"
                  f"colsample_bytree: {colsample_bytree}\n"
                  f"scale_pos_weight: {scale_pos_weight}\n"
                  f"max_depth: {max_depth}\n"
                  f"taxa de split: {split}")

        except:
            print("\nDefault configuration file not found.\nPlease check the root folder")

        training()

    elif op == 3:
        try:
            optimized = open(root_folder + "/optimized.txt", "r")
            optimizedConfig = optimized.readlines()
            print("\nOptimized file sucessfully loaded")

            eta = float(optimizedConfig[0].split(" #")[0])
            n_estimators = int(optimizedConfig[1].split(" #")[0])
            min_child_weight = int(optimizedConfig[2].split(" #")[0])
            subsample = float(optimizedConfig[3].split(" #")[0])
            colsample_bytree = float(optimizedConfig[4].split(" #")[0])
            scale_pos_weight = int(optimizedConfig[5].split(" #")[0])
            max_depth = int(optimizedConfig[6].split(" #")[0])
            split = float(optimizedConfig[7].split(" #")[0])

            print(f"Values:\n\n"
                  f"Eta: {eta}\n"
                  f"n_estimators: {n_estimators}\n"
                  f"min_child_weight: {min_child_weight}\n"
                  f"subsample: {subsample}\n"
                  f"colsample_bytree: {colsample_bytree}\n"
                  f"scale_pos_weight: {scale_pos_weight}\n"
                  f"max_depth: {max_depth}\n"
                  f"taxa de split: {split}")

            optimized.close()
            training()

        except:
            print("\nOptimized config file not found.\n"
                  "Run the optimization process through the menu first")
            input("\nPress enter to continue")
            menu()

    elif op == 4:
        optimize()

    elif op == 5:
        try:
            model = joblib.load("model.pkl")
            print("\nModel successfully loaded")

            classify(model)

        except:
            print("\nModel is not trained yet.\n"
                  "Run the training process through the menu first")
            input("\nPress enter to continue")
            menu()

    elif op == 6:

        print('\nThis tool uses the Extreme Gradient Boosting (XGB) algorithm in its construction.\n'
               'As a boosting algorithm, XGB implements several simpler algorithms in order to achieve a \n'
               'more complete and accurate result at the end.\n\n'
               'In practice, algorithms of this type work as sequential decision trees since the value that was\n'
               'predicted at n will be taken into account for the prediction at n+1 where the algorithm will use new random columns and values\n'
               'to learn to deal with the peculiarities of the WRONG classification from the previous round.\n\n'
               'Balanced Bagging Classification library was also used in this work, to deal with unbalanced data\n'
               'and Grid Search CV to make the cross-validation of parameters and values to generate the optimized values\n'
               'One of the highlights of the XGB is precisely the fact that it has dozens of configurable parameters. \n'
               'These are the main ones used by this tool:\n\n'
               'Eta: Represents the learning rate, called eta in the official XGB documentation.\n'
               'N_estimators: Refers to the number of decision trees that will be created by the model during training.\n'
               'Min_child_weight: Defines the minimum sum of weights necessary for the tree to continue to be partitioned.\n'
               'The higher this value, the more conservative the algorithm will be and the less superspecific relationships will be learned.\n'
               'Max_depth: Represents the maximum depth of the tree. Like the previous parameter, it has the same control relationship with overfitting.\n'
               'However, in this case, the higher the more likely it is to overfit.\n'
               'Subsample: Determines the portion of random training data that will be passed to each tree before they increase by another level.\n'
               'Colsample_bytree: Similar to the above, determines the portion of columns that will be given randomly when the trees are created.\n'
               'Scale_pos_weight: Controls the balance between positive and negative weights. \n'
              'It is recommended for cases with a great imbalance between classes.\n\n'
               '**************************************************** ***\n'
               '                        IMPORTANT\n\n'
              'Fill in the configuration files in the applications root folder before the first use\n'
              'for the correct functioning of the application\n\n'
              'For more information:\n'
              'Contact -> MaximilianoMeyer48@gmail.com\n'
              'For updates, clone or contribute to the project -> https://github.com/Evilmaax\n\n'
              '**************************************************** ***\n')

        x = input('Pressione enter to return to menu')
        menu()


    elif op == 7:
        x = input('\nThis application was developed by Maximiliano Meyer during his final work\n'
                   'titled "Development of a classifying tool to detect false transactions in \n'
                   'credit card operations in real-time using Machine Learning "\n'
                   'developed in the Computer Science course at the University of Santa Cruz do Sul - Unisc\n'
                   'Research and project developed in the first half of 2021.\n\n'
                   'Its use is free as long as the source is informed.\n'
                   'For updates, clone or contribute to the project -> https://github.com/Evilmaax\n'
                   'Contact -> MaximilianoMeyer48@gmail.com\n\n'
                   'Version 0.1\n\n'
                   'Press enter to return to menu')

        menu()


try:
    model = joblib.load("model.pkl")
    op = 0

    while op != 1 and op != 2:
        op = int(input("\nYou have a trained model ready to be used.\n"
                       "Do you like to:\n\n"
                       "1 - Classify transactions now\n"
                       "2 - Build model or prepare files and configuration\n\n"
                       "Answer: "))

    if op == 1:
        classify(model)

    elif op == 2:
        menu()

except:
    menu()
