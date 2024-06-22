#######updated######Working dataset for 5 documentss
##training
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_files
import fitz
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
#####for the whole one
X_train = None
X_test = None
y_train = None
y_test = None
vectorizer = None
target_names = None

def data_cleaning(text):
    """
    Method to perform data cleaning
    Removes punctuations and numbers from text

    Parameters
    ----------
    text : str
        The full corpus in the pdf/jpg.

    Returns
    -------
    str
        The cleaned text with punctuations and numbers removed from text.

    """
    punc = '''!()[]{};:'",<>./?@#$%^&*_~\\'''
    text = text.translate(str.maketrans("","", punc))
    return re.sub(r' \d\d*', "", text.lower())

# one glabal variable to define 'clf.pkl'

classifer_pickle = 'clf.pkl'




def do():
    
            
        
        




        
        

        with fitz.open("KiranMuniswappaRajanna-Purchase agreement.pdf") as doc:            
            text = ""
            for page in doc:
                text += page.get_text()


                

        ###
        string = text
        with open(classifer_pickle, 'rb') as file:
            clf = pickle.load(file)
        clf_ = clf['clf']
        model = clf['model']
        with open("vocabdata.pkl", 'rb') as f1:
            dew = pickle.load(f1)
        
        target_names=dew['target_names']
        vectorizer=dew['vectorizer']
        string = data_cleaning(text)
        str_vector = vectorizer.transform([string])
        pred = target_names[clf_.predict(str_vector)[0]]
        # use calibrated cv to calculate probabilities
        prob = model.predict_proba(str_vector)
        prob = max(prob[0])
        print(pred, prob)

do()