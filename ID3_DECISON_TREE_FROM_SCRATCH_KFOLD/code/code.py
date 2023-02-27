import numpy as np
import pandas as pd

'''READ CSV FILE, CONVERT IT NUMPY ARRAY; RETURNS 2 ARRAYS: COLUMN NAMES AND DATASET'''
def read_file(file_path):
    pandas_data_frame = pd.read_csv(file_path)
    column_names = pandas_data_frame.columns.to_numpy()
    numpy_array = pd.DataFrame(pandas_data_frame).to_numpy()
    return column_names, numpy_array


    

#------------------------------------------CALCULATING INFORMATION GAIN-------------------------------------------------
'''CALCULATE ENTROPY OF ALL DATASET; RETURNS ENTROPY OF ALL DATASET
   ARGUMENTS: DATASET= DATA FILE, LABEL INDEX = INDEX OF COLUMN WHICH IS THE ATTRIBUTE WILL BE PREDICTED'''
def get_dataset_entropy(dataset, label_index):
    #get target column
    label = dataset[:,label_index]
    #get distinct values in target column
    label_variables = np.unique(label)
    #get total number of values in target column
    total_lenght = len(label)
    
    entropy = 0
    for variable in label_variables:
        count = len(dataset[dataset[:,label_index] == variable])
        variable_entropy = -(count/total_lenght)*np.log2(count/total_lenght)
        entropy += variable_entropy
    return entropy

'''CALCULATE ENTROPY OF ATTRIBUTE, RETURNS ENTROPY OF A SPECIFIC ATTRIBUTE
   ARGUMENTS: DATASET, INDEX OF TARGET COLUMN, TARGET COLUMNS VARIABLES'''
def get_entropy(data,label_index, class_list):
    len_data = len(data)
    entropy = 0
    
    for value in class_list:
        len_sub = len(data[data[:,label_index] == value])
        sub_entropy = 0
        if len_sub != 0:
            probability = len_sub / len_data
            sub_entropy = -probability * np.log2(probability)
        entropy += sub_entropy
    return entropy

'''CALCCULATE INFORMATION GAIN OF ATTRIBUTE
   ARGUMENTS: INDEX OF ATTRIBUTE, DATASET, INDEX OF TARGET COLUMN'''
def get_information_gain(feature_index, dataset, label_index):
    #get distinc values in label column as array
    class_list = np.unique(dataset[:,label_index])
    #get distinc values in feature as array
    feature_values = np.unique(dataset[:,feature_index])
    
    number_of_rows = len(dataset)
    
    information_gain = 0.0
    
    for value in feature_values:
        feature_data = dataset[dataset[:,feature_index] == value]
        len_value = len(feature_data)
        entropy = get_entropy(feature_data, label_index, class_list)
        probability = len_value / number_of_rows
        information_gain += probability * entropy
    return get_dataset_entropy(dataset, label_index) - information_gain

'''RETURN ATTRIBUTE WHICH IS BEST CLASSIFIER, 
   ARGUMENTS DATASET, INDEX OF TARGET COLUMN'''
def get_best_classifier_index(dataset, label_index):
    best_index = 0
    max_info_gain = get_information_gain(0, dataset, label_index)
    
    for index in range(2,len(dataset[0])):
        gain = get_information_gain(index, dataset, label_index)
        if gain > max_info_gain:
            max_info_gain = gain
            best_index = index
            
    return best_index

#--------------------------------------------CREATING DECISION TREE FUNCTIONS------------------------------------------
'''function to create subtree to add under the branches
   ARGUMENTS: INDEX OF ATTRIBUTE, DATASET, INDEX OF TARGET COLUMN, VARIABLES IN TARGET COLUMN'''
def set_subtree(feature_index, dataset, label_index, class_list):
    tree = {}
    
    unique, counts = np.unique(dataset[:,feature_index],return_counts=True)
    value_count_dict = {}
    
    for u,c in zip(unique,counts):
        value_count_dict[u] = c
        
    for feature_value, count in iter(value_count_dict.items()):
        feature_value_data = dataset[dataset[:,feature_index] == feature_value]
        
        assigned_to_node = False
        
        for c in class_list:
            len_class = len(feature_value_data[feature_value_data[:,label_index] == c ])
            
            if len_class == count:
                tree[feature_value] = c
                dataset = dataset[dataset[:,feature_index] != feature_value]
                assigned_to_node = True
                
        if not assigned_to_node:
            tree[feature_value] = "?"
    return tree, dataset

'''recursive function to create decision tree
   ARGUMENTS: TREE DICTIONARY, NODE VALUE, DATASET, TARGET COLUMN INDEX, VARIABLES IN TARGET COLUMN'''
def set_tree(root, prev_value, dataset, label_index, class_list):
    if len(dataset) != 0:
        best_classifier_index = get_best_classifier_index(dataset, label_index)
        tree, dataset = set_subtree(best_classifier_index, dataset, label_index, class_list)
        next_root = None
        
        if prev_value != None:
            root[prev_value] = dict()
            root[prev_value][best_classifier_index] = tree
            next_root = root[prev_value][best_classifier_index]
        else:
            root[best_classifier_index] = tree
            next_root = root[best_classifier_index]
            
        for node, branch in list(next_root.items()):
            if branch == "?":
                feature_value_data = dataset[dataset[:,best_classifier_index] == node]
                set_tree(next_root, node, feature_value_data, label_index, class_list)
        
'''calls tree functions and return decision tree
   ARGUMENTS: TRAIN DATA, INDEX OF TARGET COLUMN'''
def ID3(train_data, label_index):
    dataset = train_data.copy()
    tree = {}
    class_list = np.unique(dataset[:,label_index])
    set_tree(tree, None, dataset, label_index, class_list)
    return tree

#-------------------------------------------ID3 CLASSIFICATION-------------------------------------------------------
'''RETURNS: PREDICTION OF TEST POINT
   ARHUMENTS: TREE, TEST POINT'''
def ID3_Prediction(tree, test_point):
    if not isinstance(tree, dict):
        return tree
    else:
        root_node = next(iter(tree))
        feature_value = test_point[root_node]
        if feature_value in tree[root_node]:
            return ID3_Prediction(tree[root_node][feature_value], test_point)
        else:
            return None

'''RETURNS PREDICTION VALUES OF DATA POINTS IN TEST SET
   ARGUMENTS: TRAIN SET, TEST SET, INDEX OF TARGET COLUMNS
   RETURNS: PREDICTIONS AND DECISION TREE'''        
def ID3_Classification(train_set, test_set, label_index):
    tree = ID3(train_set, label_index)
    
    predictions = np.empty(shape=(len(test_set)),dtype = object)
    for index in range(len(test_set)):
        predictions[index] = ID3_Prediction(tree, test_set[index])
        
    return predictions , tree


#-------------------------------------------DISCREATIZARION-----------------------------------------------------------
'''FIND BEST VALUE FOR INTERVAL IN COLUMN BASED ON INFIRMATION GAIN CALCULATED WITH EACH VALUE IN COLUMN
   ARGUMENTS: INDEX OF ATTRIBUTE , DATASET, INDEX OF TARGET COLUMN
   RETURNS COLUMN THAT CHANGED NUMERICAL TO INTERVAL'''
def convert_interval(feature_index, dataset, label_index):
    
    column = dataset[:,feature_index]
    best_ig = -10
    best_interval_column = np.empty(shape=(len(column)),dtype = object)
    for integer in column:
        
        new = np.empty(shape=(len(column)),dtype = object)
        sub1 = column < integer
        sub2 = column >= integer
        
        new[sub1] = " < " + str(integer)
        new[sub2] = " >= " + str(integer)


        temp_dataset = dataset.copy()
        temp_dataset[:,feature_index] = new
        
        ig = get_information_gain(feature_index, temp_dataset, label_index)
        if ig > best_ig:
            best_ig = ig
            best_interval_column = new
            
    return best_interval_column
'''CHANGE NUMERICAL VALUES TO DISCRETE VALUES
   ARGUMENTS: DATASET, INDEX OF TARGET COLUMN
   RETURNS DATASET THAT NUMERICAL VALUES CHANGED TO INTERVALS'''
def discreate(dataset, label_index):
    
    for index in range(len(dataset[0])):
        if (type(dataset[:,index][0]) != str):
            dataset[:,index] = convert_interval(index, dataset, label_index)
            
    return dataset
#-------------------------------------------K FOLD FUNCTIONS---------------------------------------------------------
'''SPLITS DATASET TO N FOLDS
   ARGUMENTS: DATASET,FOLD NUMBER
   RETURN N FOLDS DATSET'''
def split(dataset, fold_number):
    fold_size  = int(len(dataset)/5)
    col_size = int(len(dataset[0]))
    split = np.empty(shape=(5,fold_size,col_size),dtype = object)
    
    dataset_copy = dataset.copy()
    
    #create array contains randomized values between range 0 to len(dataset)
    #get randomized data
    random_indexes = np.arange(len(dataset))
    np.random.shuffle(random_indexes)
    
    x = 0
    y = fold_size
    
    for z in range(5):
        current_indexes = random_indexes[x:y]
        
        split[z] = dataset_copy[current_indexes]
        
        x += fold_size
        y += fold_size
          
    return split

'''EVALUATES ALGORITHMS AND METRICS WITH K-FOLD METHOD
   ARGUMENTS: DATASET, ALGORITHM, METRIC, LABEL OF TARGET COLUMNS 
   RETURNS METRIC SCORES OF EACH FOLD AND DECISION TREE OF BEST SCORE'''
def KFOLD(dataset, algorithm, metric, label_index):
    folds = split(dataset, 5)
    scores = np.zeros(shape=(5))
    best_tree = {}
    best_score = -10
    for index in range(len(folds)):
        train_set = folds.copy()
        train_set = np.delete(train_set,index,axis=0)
        test_set = folds[index].copy()
        
        #reshape train set
        row_shape = (4) * len(folds[index])
        column_shape = len(folds[index][0])
        train_set = np.reshape(train_set,(row_shape,column_shape))
        
        actual = test_set[:,label_index]
        
        predicted , tree = algorithm(train_set, test_set, label_index)
        
        confusion_matrix = create_confusion_matrix(actual, predicted)
        
        scores[index] = metric(confusion_matrix)
        
        if scores[index] > best_score:
            best_score = scores[index]
            best_tree = tree
        
        
    return scores , best_tree


#------------------------------------CONFUSION MATRIX AND PERFORMANCE METRICS----------------------------------------
'''CREATES CONFUSION MATRIX BASED ON ACTUAL TEST DATA VALUES AND PREDICTION VALUES
   ARGUMENTS: ACTUAL VALUES OF TEST DATA, PREDICTED VALUES OF TEST DATA
   RETURNS NXN CONFUSION METRICS'''
def create_confusion_matrix(actual,predicted):
    confusion_matrix = np.zeros(shape=(2,2))
    row = 0
    column = 0
    
    for index in range(len(actual)):
        if actual[index] == "Yes":
            row = 0
        if actual[index] == "No":
            row = 1
        if predicted[index] == "Yes":
            column = 0
        if predicted[index] == "No":
            column = 1
            
        confusion_matrix[row,column] += 1
    return confusion_matrix
'''CALCULATES ACCURACY SCORE FROM CONFUSION MATRIX
   ARGUMENTS: CONFUSION MATRIX
   RETURNS: ACCURACY'''
def accuracy(confusion_matrix):
    TP = confusion_matrix[0][0]
    TN = confusion_matrix[1][1]
    
    return (TP + TN) / np.sum(confusion_matrix)
'''CALCULATES PRECISION SCORE FROM CONFUSUION MATRIX
   ARGUMENTS: CONFUSUION MATRIX'''
def precision(confusion_matrix):
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[1][0]
    
    return TP / (TP + FP)
'''CALCULATES RECALL FROM CONFUSION MATRIX'''
def recall(confusion_matrix):
    TP = confusion_matrix[0][0]
    FN = confusion_matrix[0][1]
    
    return TP / (TP + FN)
'''CALCULATES F1 SCORE FROM CONFUSION MATRIX'''
def f1_score(confusion_matrix):
    RECALL = recall(confusion_matrix)
    PRECISION = precision(confusion_matrix)
    
    return (2*(RECALL*PRECISION)) / (RECALL + PRECISION)



'''PRINTS DICTIONARY OF DECISION TREE'''
def print_tree(tree,names,nesting=-5):
    
    if type(tree) == dict:
        print('')
        nesting += 5
        for k in tree:
            print(nesting* ' ',end = '')
            if type(k) == int:
                print(names[k],end=":")
            else:
                print(k,end=":")
            print_tree(tree[k],names, nesting)
    else:
        print(tree)

