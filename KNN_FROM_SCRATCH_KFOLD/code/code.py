import numpy as np
import pandas as pd #used just for reading data

np.set_printoptions(suppress=True)

#---------------------------------------FUNNCTIONS TO READ AND PREPARE DATASET------------------------------------------
#read data with pandas and convert it to numpy array
def read_csv(file_path):
    pandas_data_frame = pd.read_csv(file_path)
    numpy_array = pd.DataFrame(pandas_data_frame).to_numpy()
    return numpy_array

#drop unnecessary column
def delete_first_column(dataset):
    return np.delete(dataset,0,axis=1)
    

'''these two functions for part1 dataset.It gives specific integer value to class string name and converts
strings to these specific integer values for e.g INTP = 0 or ISTP = 1'''
#convert string column to integer
def str_to_integer(dataset,column):
    class_names = [row[column] for row in dataset]
    unique = set(class_names)
    lookup = dict()
    for i,value in enumerate(unique):
        lookup[value] = i
        #print('[%s] -> %d ' % (value, i))
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

#mapping class names with integers and change them to integers
def map_class_names(dataset):
    row,column = dataset.shape
    mapping_class_names = str_to_integer(dataset,column-1 )
    print("converted as : ",mapping_class_names)
    return mapping_class_names    
    

#-------------------------------------------------KNN-CLASSIFICATION---------------------------------------------------

'''ARGUMENTS: TRAIN DATA TEST POINT NUMBER OF NEIGHBOR
   RETURNS PREDICTION CLASS OF DATA POINT 
   CALCULATES DISTANCES, GETS K NEAREST NEIGHBORS, RETURNS MODE OF NEIGHBORS CLASS VALUES'''
def classification(train, test_row, k):
    #get distances between rows[:-1]
    data = np.delete(train,-1,axis=1).astype(float)
    test = test_row[:-1].astype(float)
    distances = np.linalg.norm(data - test,axis=1)
   
    #map index with distances 
    indexed = np.arange(len(train),dtype=float).reshape(len(train),1)
    indexed = np.insert(indexed, 1, distances,axis=1)
    
    #sort by distances
    indexed = indexed[np.argsort(indexed[:,len(indexed[0])-1])]
   
    #get k nearest neighbors
    indexed = indexed[:k]
    indexes = indexed[:,0].astype(int)
    
    #get mode of class outcomes
    classes = train[indexes][:,-1].astype(int)
    
    return np.bincount(classes).argmax()
    
    
'''ARGUMENTS: TRAIN DATA, TEST DATA , K
   RETURNS PREDICTIONS SET 
   CALCULATES DISTANCE OF EACH DATA IN TEST DATA BETWEEN EACH DATA IN TRAIN DATA 
   RETURNS PREDICTION OF EACH DATA IN TEST SET'''
def KNN_classification(train_set, test_set, k):
    predictions = np.empty(shape=(len(test_set)))
    for index in range(len(test_set)):
        predictions[index] = classification(train_set, test_set[index], k)
        
    return predictions
#----------------------------------------------KNN REGRESSION-----------------------------------------------------------
def regression(train,test_row,k):
    #get distances between rows[:-1]
    data = np.delete(train,-1,axis=1).astype(float)
    test = test_row[:-1].astype(float)
    distances = np.linalg.norm(data - test,axis=1)
   
    #map index with distances 
    indexed = np.arange(len(train),dtype=float).reshape(len(train),1)
    indexed = np.insert(indexed, 1, distances,axis=1)
    
    #sort by distances
    indexed = indexed[np.argsort(indexed[:,len(indexed[0])-1])]
   
    #get k nearest neighbors
    indexed = indexed[:k]
    indexes = indexed[:,0].astype(int)
    
    #get mode of class outcomes
    classes = train[indexes][:,-1].astype(int)
    
    return np.sum(classes)/len(classes)

def KNN_regression(train_set,test_set,k):
    predictions = np.empty(shape=(len(test_set)))
    for index in range(len(test_set)):
        predictions[index] = regression(train_set, test_set[index], k)
    return predictions
#-----------------------------------WEIGHTED KNN------------------------------------------------------------------------
def weighted_classification(train,test_row,k):
    #get distances between rows[:-1]
    data = np.delete(train,-1,axis=1).astype(float)
    test = test_row[:-1].astype(float)
    distances = np.linalg.norm(data - test,axis=1)
    weights = 1 / distances
    
    #map index with distances 
    indexed = np.arange(len(train),dtype=float).reshape(len(train),1)
    indexed = np.insert(indexed, 1, distances,axis=1)
    indexed = np.insert(indexed, 1, weights,axis=1)
    
    
    #sort by distances
    indexed = indexed[np.argsort(indexed[:,len(indexed[0])-1])]
   
    #get k nearest neighbors
    indexed = indexed[:k]
    classes = train[indexed[:,0].astype(int)][:,-1].copy()
    indexed[:,0] = classes
    
    class_weights = np.zeros(shape=(16))
    for x in indexed:
        z = int(x[0])
        class_weights[z] += x[1]
        
    return np.argmax(class_weights)

def weighted_knn_classification(train_set,test_set,k):
    predictions = np.empty(shape=(len(test_set)))
    for index in range(len(test_set)):
        predictions[index] = weighted_classification(train_set, test_set[index], k)
    return predictions

#--------------------------------------WEIGHTED KNN REGRESSION----------------------------------------------------------


#------------------------------------------------K-FOLD CROSS VALIDATION------------------------------------------------
'''N X M TO 5X N/5 X M ARRAY'''
def split(dataset, n=5):
    fold_size  = int(len(dataset)/5)
    col_size = int(len(dataset[0]))
    split = np.empty(shape=(5,fold_size,col_size))
    
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
        

        
        
def KFOLD(dataset, algorithm, metric, k ):
    folds = split(dataset)
    scores = np.zeros(shape=(5))
    cm = np.empty(shape=(5,16,16))
    for index in range(len(folds)):
        train_set = folds.copy()
        train_set = np.delete(train_set,index,axis=0)
        test_set = folds[index].copy()
        
        #reshape train set
        row_shape = (4) * len(folds[index])
        column_shape = len(folds[index][0])
        train_set = np.reshape(train_set,(row_shape,column_shape))
        
        actual = folds[index][:,-1].copy()
        predicted = algorithm(train_set, test_set, k)
        
        confusion_matrix = create_confusion_matrix(actual, predicted)
        
        cm[index] = confusion_matrix
        scores[index] = metric(confusion_matrix)
    
    return scores
    
def KFOLD_REGRESSION(dataset,algorithm,k):
    folds = split(dataset)
    scores = np.zeros(shape=(5))
    
    for index in range(len(folds)):
        train_set = folds.copy()
        train_set = np.delete(train_set,index,axis=0)
        test_set = folds[index].copy()
        
        #reshape train set
        row_shape = (4) * len(folds[index])
        column_shape = len(folds[index][0])
        train_set = np.reshape(train_set,(row_shape,column_shape))
        
        actual = folds[index][:,-1].copy()
        predicted = algorithm(train_set, test_set, k)
        
        
        scores[index] = MAE(actual,predicted)
    return scores
        
#---------------------------------------------CREATE CONFUSION MATRIX---------------------------------------------------
def create_confusion_matrix(actual,predicted):
    confusion_matrix = np.zeros(shape=(16,16))
    
    for index in range(len(actual)):
        row = int(actual[index])
        column = int(predicted[index])
        
        confusion_matrix[row,column] += 1
        
    return confusion_matrix

#--------------------------------------------PERFORMANCE METRICS--------------------------------------------------------
def accuracy(confusion_matrix):
    class_scores = np.zeros(shape=(16))
    
    for index in range(16):
        TP = confusion_matrix[index][index]
        FP = np.sum(confusion_matrix,axis=0)[index] - TP
        FN = np.sum(confusion_matrix,axis=1)[index] - TP
        TN = np.sum(confusion_matrix) - (TP+FP+FN)
        
        class_scores[index] = (TP+TN) / np.sum(confusion_matrix)
        
    return np.sum(class_scores) / 16

def precision(confusion_matrix):
    class_scores = np.zeros(shape=(16))
    
    
    for index in range(16):
        TP = confusion_matrix[index][index]
        FP = np.sum(confusion_matrix,axis=0)[index] - TP
        
        class_scores[index] = TP / (TP+FP)
        
    return np.sum(class_scores) / 16

def recall(confusion_matrix):
    class_scores = np.zeros(shape=(16))
   
    
    for index in range(16):
        TP = confusion_matrix[index][index]
        FN = np.sum(confusion_matrix,axis=1)[index] - TP
        
        class_scores[index] = TP / (TP+FN)
        
    return np.sum(class_scores) / 16

#-------------------------------------MIN MAX NORMALIZATION-------------------------------------------------------------
#find minimun and maximum values for each column in dataset
def get_min_max(dataset):
    min_max = np.empty(shape=(len(dataset[0])-1,2))
    
    mins = dataset.min(axis=0)
    maxs = dataset.max(axis=0)
    
    for i in range(len(dataset[0])-1):
        min_max[i] = [mins[i], maxs[i]]


    return min_max

#normalize colums except class labels
def min_max_normalization(dataset):
    min_max_list = get_min_max(dataset)
    
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - min_max_list[i][0]) / (min_max_list[i][1] - min_max_list[i][0])
            
    return dataset

#------------------------------------MEAN ABS ERROR--------------------------------------------------------------------
def MAE(actual_values, predicted_values):
    absolute_errors = np.empty(shape=(len(actual_values)))
    for i in range(len(actual_values)):
        absolute_errors[i] = np.absolute(actual_values[i] - predicted_values[i])
        
    return np.sum(absolute_errors) / len(absolute_errors)
#---------------------------------------TEST-----------------------------------------------------------------------------

