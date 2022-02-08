import os
import sys
import pandas as pd
import nltk
import string
from nltk.stem.porter import *
from nltk.corpus import stopwords


def trainNaiveBayes(train_data, stop_word_rm = False, stem_rm = False):
    
    
    word_in_class_count = {}
    all_true = []
    all_lie = []
    word_in_class_probability = {}
    
    # remove punctuation, tokenize
    tokenized_list = []
    exclude = set(string.punctuation)
    for index, row in train_data.iterrows():
        punct_removed = ''.join(ch for ch in row['statement'] if ch not in exclude)
        tokenList = nltk.word_tokenize(punct_removed)
        tokenList = [word.lower() for word in tokenList]
        if stem_rm:
            p_stemmer = PorterStemmer()
            tokenList = [p_stemmer.stem(word) for word in tokenList]
            
        if stop_word_rm:
            nltk_stop_words = set(stopwords.words("english"))
            tokenList = [w for w in tokenList if w not in nltk_stop_words]
        tokenized_list.append(tokenList)
    if 'tokenizedList' not in train_data:
        train_data.insert(1, "tokenizedList", tokenized_list, False)
        
    # creat a dictionary in which the keys are our vocabulary and values are the count of the words in each 
    # class
    for index, row in train_data.iterrows():
        for word in row['tokenizedList']:
            if word not in word_in_class_count:
                if row['class'] == 'lie':
                    word_in_class_count[word] = {'lie': 1, 'true': 0}
                elif row['class'] == 'true':
                    word_in_class_count[word] = {'lie': 0, 'true': 1}
            else:
                if row['class'] == 'lie':
                    word_in_class_count[word]['lie'] += 1
                elif row['class'] == 'true':
                    word_in_class_count[word]['true'] += 1
    # number of words in class true
    for index, row in train_data.iterrows():
        for word in row['tokenizedList']:
            if row['class'] == 'lie':
                if word not in all_lie:
                    all_lie.append(word)
            elif row['class'] == 'true':
                if word not in all_true:
                    all_true.append(word)
    
    # word in class probability, add-1 smoothing
    V = len(word_in_class_count)
    for word in word_in_class_count:
        word_in_class_probability[word] = {'lie': (word_in_class_count[word]['lie'] + 1)/(len(all_lie)+V), 
                                          'true': (word_in_class_count[word]['true'] + 1)/(len(all_true)+V)}
    
    # probability of each class
    lie_count = 0
    true_count = 0
    for index, row in train_data.iterrows():
        if row['class'] == 'lie':
            lie_count+=1
        elif row['class'] == 'true':
            true_count+=1
    lie_prob = lie_count / (lie_count + true_count)
    true_prob = true_count / (lie_count + true_count)
    
    return word_in_class_probability, lie_prob, true_prob
    
    
    
def testNaiveBayes(nbClassifier, test_statement, stop_word_rm = False, stem_rm = False):
    exclude = set(string.punctuation)
    punct_removed = ''.join(ch for ch in test_statement if ch not in exclude)
    tokenList = nltk.word_tokenize(punct_removed)
    tokenList = [word.lower() for word in tokenList]
    if stem_rm:
        p_stemmer = PorterStemmer()
        tokenList = [p_stemmer.stem(word) for word in tokenList]

    if stop_word_rm:
        nltk_stop_words = set(stopwords.words("english"))
        tokenList = [w for w in tokenList if w not in nltk_stop_words]
        
    # probability for lie class
    p_lie = 1
    for word in tokenList:
        p_lie *= nbClassifier[word]['lie']
    
    p_true = 1
    for word in tokenList:
        p_true *= nbClassifier[word]['true']
    
    if p_lie > p_true:
        return False
    else:
        return True
        
def main():
    path = os.getcwd()
    rows = []
    columns = ['statement', 'fileName', 'class']
    if len(sys.argv) > 2:
        files_path = sys.argv
    else:
        files_path = 'bestfriend.deception.training'
    for roots, dirs, files in os.walk(path+ '/' + files_path):
        for file in files:
        #rows.append()
            if file.split()[0].startswith('lie'):
                doc = open(roots+'/'+file, 'r')
                doc_str = doc.read()
                rows.append([doc_str, file, 'lie'])
                doc.close()
            elif file.split()[0].startswith('true'):
                doc = open(roots+'/'+file, 'r')
                doc_str = doc.read()
                rows.append([doc_str, file,'true'])
                doc.close()

    training_data = pd.DataFrame(rows, columns = columns)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    fileName_class = []

    for i in range(len(training_data)):
        train = training_data.iloc[0:i]
        train2 = training_data.iloc[i+1:len(training_data)]
        train = pd.concat([train,train2])
        test = training_data.iloc[i]['statement']
        check = training_data.iloc[i]['class']
        fname = training_data.iloc[i]['fileName']
        classifier = trainNaiveBayes(training_data.copy())[0]
        pred_class = testNaiveBayes(classifier, test)
        fileName_class.append((fname, pred_class))
        if pred_class == True and check == 'true':
            tp += 1
        elif pred_class == True and check == 'lie':
            fp += 1
        elif pred_class == False and check == 'lie':
            tn += 1
        elif pred_class == False and check == 'true':
            fn += 1

    print(fileName_class)
    print(f'accuracy: {(tp + tn)/ (tp+fp+fn+tn)}')

    probs = trainNaiveBayes(training_data.copy())[0]
    print('Top 10 from class true:')
    sorted_true = sorted(probs, key=lambda x:probs[x]['true'], reverse = True)
    print(sorted_true[:10])
    sorted_true = sorted(probs, key=lambda x:probs[x]['true'], reverse = False)
    print(sorted_true[:10])

    print('Top 10 from class lie:')
    sorted_lie = sorted(probs, key=lambda x:probs[x]['lie'], reverse = True)
    print(sorted_lie[:10])
    sorted_lie = sorted(probs, key=lambda x:probs[x]['lie'], reverse = False)
    print(sorted_lie[:10])

if __name__ == '__main__':
	main()

