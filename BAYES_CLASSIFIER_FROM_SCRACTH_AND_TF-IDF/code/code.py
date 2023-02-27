import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

'''read data with pandas and convert it to numpy array
   arguments: file path or file name
   returns: numpy data array'''
def read_csv(file_path):
    pandas_data_frame = pd.read_csv(file_path)
    numpy_array = pd.DataFrame(pandas_data_frame).to_numpy()
    return numpy_array

'''map target string names to integer values    will be used in str_to_integer function
   arguments: dataset and index of target column
   returns map of string to integers'''
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

'''change string names of target labels to integer values (e.g. sports = 1 ets)
   arguments: dataset
   returns nothing just changes string names of labels'''
def map_class_names(dataset):
    row,column = dataset.shape
    mapping_class_names = str_to_integer(dataset,column-1 )
    print("converted as : ",mapping_class_names)
    return mapping_class_names

'''remove unnecessary symbols from texts
   return dataset'''
def initialize_text(dataset):
    texts = dataset[:,1]
    for index in range(len(texts)):
    
        texts[index] = texts[index].replace(" - ", " ")
        texts[index] = texts[index].replace(", "," ")
        texts[index] = texts[index].replace(":","")
        texts[index] = texts[index].replace(".","")
        texts[index] = texts[index].replace(" s "," ")
        texts[index] = texts[index].replace("don t","dont")
        
    dataset[:,1] = texts
    return dataset

'''remove stop keywords and return dataset'''
def remove_stop_words(dataset):
    
    copy = dataset.copy()
    texts = copy[:,1]
    
    for index in range(len(texts)):
        texts[index] = np.array(texts[index].split())
        texts[index] = [word for word in texts[index] if not word in ENGLISH_STOP_WORDS]
        texts[index] = (" ").join(texts[index])
    copy[:,1] = texts
    return copy

#--------------------------------------------------BAG OF WORDS FUNCTIONS------------------------------------------------
'''get most frequent words in vocabulary
   threshold is the number of how many most frequent words you want to use in bag of words'''
def alter_histogram(histogram, threshold):
    filtered_histogram = histogram[0:threshold]
    return filtered_histogram


'''calculate probablity with laplace smoothing'''
def laplace_smoothing(number_of_wi, number_of_total, alpha,number_of_classes):
    
    return (number_of_wi + alpha) / (number_of_total + alpha*number_of_classes)

'''convert text data to text array, each element in array is a word'''
def get_unigram_array(text):
    #convert text to numpy array , each element in array is word of text
    text_array = np.array(text.split())
    return text_array

'''convert text data to text array, each element in array is double words pairs in texts'''
def get_bigram_array(text):
    #convert text to words array
    text_array = np.array(text.split())
    
    #convert array to double word pairs
    double = np.empty(shape=(len(text_array)-1),dtype = object)
    
    index = 0
    next_index = 1
    
    while next_index < len(text_array):
        double[index] = text_array[index] + " " + text_array[next_index]
        index += 1
        next_index += 1
    return double


'''create unigram vocabulary histogram from all texts in dataset
   returns unique words - frequencies histogram in descending sorted order'''
def get_unigram_histogram(dataset):
    
    #get all text data
    texts = dataset[:,1]
    
    #combine texts into a single text data
    aux = " ".join(texts)
    
    #convert text to numpy array , each element in array is word of text
    text = get_unigram_array(aux)
    
    #get the unique words in text and get their counts
    words, counts = np.unique(text,return_counts=True)
    
    #create histogram that holds words - counts values
    histogram = np.empty(shape=(len(words),2),dtype=object)
    #first column pf histogram is frequencies and second column is words
    histogram[:,0] = words
    histogram[:,1] = counts
    
    #sort histogram in descending order
    histogram = histogram[np.argsort(histogram[:,1])]
    histogram = np.flip(histogram)
    return histogram 

'''create bigram vocabulary from given text data
   returns words pair in dataset and their counts'''
def get_bigram_histogram(dataset):
    
    #get all tex data
    texts = dataset[:,1]
    #combine texts into single text data
    aux = " ".join(texts)
    #convert text to numpy array , each element in array is double word pairs of text
    double = get_bigram_array(aux)
    
    #get unique word pairs and their counts
    words , counts = np.unique(double,return_counts=True)
    
    #create histogram
    histogram = np.empty(shape=(len(words),2),dtype=object)
    
    #first column pf histogram is frequencies and second column is words
    histogram[:,0] = words
    histogram[:,1] = counts
    
    #sort histogram in descending order
    histogram = histogram[np.argsort(histogram[:,1])]
    histogram = np.flip(histogram)
    return histogram


'''create matrix where the columns represents the most frequent words in vocabulary(histogram)
   and rows represents texts in datasets. Returns vectorized text values with shape len_dataset x len histogram
   last column of matrix represents label values'''
def get_bow_vectors_unigram(dataset , histogram):
    vectors = np.zeros(shape=(len(dataset),len(histogram)))
    
    for index in range(len(dataset)):
        text = dataset[index][1]
        text_array = get_unigram_array(text)
        
        words, counts = np.unique(text_array,return_counts=True)
        for x in range(len(words)):
            word = words[x]
            value = counts[x]
            
            indexed = histogram[:,1] == word
            
            vectors[index][indexed] = value
    #add labels to vector table
    labels = dataset[:,-1]

    vectors = np.insert(vectors, len(histogram), labels, axis=1)
    return vectors

'''create matrix where the columns represents the most frequent words in vocabulary(histogram)
   and rows represents texts in datasets. Returns vectorized text values with shape len_dataset x len histogram
   last column of matrix represents label values'''
def get_bow_vectors_bigram(dataset, histogram):
    vectors = np.zeros(shape=(len(dataset),len(histogram)))
    
    for index in range(len(dataset)):
        text = dataset[index][1]
        text_array = get_bigram_array(text)
        
        words, counts = np.unique(text_array,return_counts=True)
        for x in range(len(words)):
            word = words[x]
            value = counts[x]
            
            indexed = histogram[:,1] == word
            
            vectors[index][indexed] = value
    #add labels to vector table
    labels = dataset[:,-1]

    vectors = np.insert(vectors, len(histogram), labels, axis=1)
    return vectors
    

#---------------------------------------------NAIVE BAYES CLASSIFICATION---------------------------------------------
'''create naive bayes based on vectors array created with create_bag_of_words function 
   returns prediction of output of new data, 
   unigram_or_bigram 1 = unigram
   unigram_or_bigram 0 = bigram '''
def naive_bayes(vectors,histogram, new_data, alpha , unigram_or_bigram = 0):
    classes = np.unique(vectors[:,-1])
    probabilities = np.zeros(shape=(len(classes),2))
    #get each word in text
    
    new_text = new_data[1]
    words = get_bigram_array(new_text)
    if unigram_or_bigram == 1:
        words = get_unigram_array(new_text)
        
    for index in range(len(classes)):
        probability = 0.0 
        
        subset = vectors[vectors[:,-1] == classes[index]]
        number_of_matched = len(subset)
        number_of_total = len(vectors)
        
        P_class = laplace_smoothing(number_of_matched, number_of_total, alpha, len(classes))
        
        for word in words:
            #find index in vectors where column of vectors represent word
            position = np.where(histogram[:,1] == word)
            
            if not np.any(position):
                
                p_word = laplace_smoothing(0, 0, alpha, len(classes))
                probability += np.log(p_word)
            else:
                n_total = np.sum(vectors[:,position])
                n_matched = np.sum(subset[:,position])
                
                p_word = laplace_smoothing(n_matched, n_total, alpha, len(classes))
                
                probability += np.log(p_word)
            
        probability += np.log(P_class)
        probabilities[index][0] = probability
        probabilities[index][1] = classes[index]
        
    
    probabilities = probabilities[np.argsort(probabilities[:,0])]
    return int(probabilities[:,1][-1])

''' 1 for unigram 0 for bigram'''
def naive_bayes_algorithm(vectors, histogram ,test, unigram_or_bigram):
    
    predictions = np.empty(shape=(len(test)))
    
    if unigram_or_bigram == 1:
        for index in range(len(test)):
            predictions[index] = naive_bayes(vectors, histogram, test[index], 1, 1)
            
    else:
        for index in range(len(test)):
            predictions[index] = naive_bayes(vectors, histogram, test[index], 1, 0)
            
    return predictions
            
        
def accuracy(real , predicted):
    vals = real == predicted
    correct = np.count_nonzero(vals)
    
    return 100 * (correct / len(vals))

#--------------------------------------------TF IDF------------------------------------------------------------------
'''1 for unigram 0 for bigram'''
def create_TF(dataset,histogram, uni_or_bi ):
    tf_table = np.zeros(shape=(len(histogram),len(dataset)))
    
    if uni_or_bi == 1:
        for index in range(len(dataset)):
            sentence = dataset[index][1]
            sentence_array = get_unigram_array(sentence)
            
            number_of_total_words = len(sentence_array)
            
            words, counts = np.unique(sentence_array,return_counts=True)
            for x in range(len(words)):
                count = counts[x]
                freq = count / number_of_total_words
                
                y = histogram[:,1] == words[x]
                
                tf_table[:,index][y] = freq
    if uni_or_bi == 0:
        for index in range(len(dataset)):
            sentence = dataset[index][1]
            sentence_array = get_bigram_array(sentence)
            
            number_of_total_words = len(sentence_array)
            
            words, counts = np.unique(sentence_array,return_counts=True)
            for x in range(len(words)):
                count = counts[x]
                freq = count / number_of_total_words
                
                y = histogram[:,1] == words[x]
                
                tf_table[:,index][y] = freq
            
            
    return tf_table
        
def create_IDF_unigram(dataset,histogram):
    number_of_sentences = len(dataset)
    
    df_table = np.zeros(shape=(len(histogram),1))
    
    words = histogram[:,1]
    
    texts = dataset[:,1].astype(str)
    
    for index in range(len(words)):
        
        vals = np.char.find(texts, words[index])
        
        found = len(vals) - np.count_nonzero(vals, -1)
        
        value = np.log10( (1+number_of_sentences) / (1+ found))
        
        df_table[index] = value
        
    return df_table

def create_TF_IDF_unigram(dataset, histogram, uni_or_bi):
    tf_matrix = create_TF(dataset, histogram,  uni_or_bi)
    df_matrix = create_IDF_unigram(dataset , histogram)
    
    matrix = tf_matrix * df_matrix
    
    
    return matrix.transpose()


    
