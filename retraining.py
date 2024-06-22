import re
import pickle
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


# Initialize variables
X_train = None
X_test = None
y_train = None
y_test = None
vectorizer = None
target_names = None

# Define a function to clean text data
def data_cleaning(text):
    punc = '''!()[]{};:'",<>./?@#$%^&*_~\\'''
    text = text.translate(str.maketrans("", "", punc))  # remove punctuation
    return re.sub(r' \d\d*', "", text.lower())  # remove digits


# Define a function to load and clean the dataset
def loading_dataset(input_folder):
    # Load the dataset from files in a folder
    dataset = load_files(input_folder, encoding="ISO-8859-1", shuffle=True)
    # Clean the text data in the dataset
    for index, data in enumerate(dataset.data):
        dataset.data[index] = data_cleaning(data)
    
    # Get the target class names from the dataset
    target_names = dataset.target_names
    print(target_names)
    
    # Save the cleaned dataset to a pickle file
    with open('vocabdata.pkl', 'wb') as vocab:
        pickle.dump(dataset, vocab)
    
    # Return the target class names
    return target_names

def training_parameters():
    with open("vocabdata.pkl", 'rb') as f1:
        dataset = pickle.load(f1)
    X_train, X_test, y_train, y_test = \
        train_test_split(dataset.data, dataset.target, test_size=0.3,
                         random_state=42, stratify=dataset.target)
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.7,
                                 min_df=1, stop_words="english",
                                 lowercase=True, ngram_range=(1, 3))
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, y_train, y_test, vectorizer, dataset.target_names

# Set the filename for the saved classifier object
classifer_pickle = 'clf.pkl'

def training_model(X_train, X_test, y_train, y_test, vectorizer):
    # specify the hyperparameters to tune for SGDClassifier
    tuned_parameters = {
        'loss': ['hinge'],
        'penalty': ['l2', 'l1'],
        'alpha': [10 ** x for x in range(-6, 1)]
    }

    # initialize SGDClassifier with some parameters
    clf = SGDClassifier(alpha=0.0001, max_iter=100, penalty='l2',
                        class_weight="balanced")

    # perform grid search with cross-validation to tune the hyperparameters
    clf = GridSearchCV(clf, tuned_parameters, cv=5)
    clf.fit(X_train, y_train)

    # evaluate the performance of the model on test data
    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    # calibrate the classifier and fit on training data
    model = CalibratedClassifierCV(clf)
    model.fit(X_train, y_train)

    # print classification report
    print("classification report:")
    print(metrics.classification_report(y_test, pred, target_names=target_names))

    # save the trained model, the vectorizer, and training data for later use
    data = {'clf': clf, 'model': model, 'X_train': X_train,
            'y_train': y_train}

    with open(classifer_pickle, 'wb') as file2:
        pickle.dump(data, file2)



def testing():

    # Open the PDF file and extract its text
    with fitz.open("/home/shanto/Sample Documents/Solar/9. Solar Agreement.pdf") as doc:            
        text = ""
        for page in doc:
            text += page.get_text()

    # Load the preprocessed data and trained classifier
    with open("vocabdata.pkl", 'rb') as f1:
        dataset = pickle.load(f1)        
    with open(classifer_pickle, 'rb') as file:
        clf = pickle.load(file)

    # Extract features from the input text using the same vectorizer used during training
    string = data_cleaning(text)
    str_vector = vectorizer.transform([string])

    # Use the trained classifier to predict the category of the input text
    clf_ = clf['clf']
    model = clf['model']
    pred = dataset.target_names[clf_.predict(str_vector)[0]]

    # Use calibrated cv to calculate the probability of the predicted category
    prob = model.predict_proba(str_vector)
    prob = max(prob[0])

    # Print the predicted category and its probability
    print(pred, prob)



#location of the dataset
input_folder = "/home/tirzok/data3"

# load the dataset and clean the data
loading_dataset(input_folder)

# extract the features from the dataset and split into training and testing sets
X_train, X_test, y_train, y_test, vectorizer, target_names = training_parameters()

# train the model using the training data and evaluate on the test data
training_model(X_train, X_test, y_train, y_test, vectorizer)

#Test a document to check accuracy and prediction.
#remove the hash of the following line to test.
#testing()





