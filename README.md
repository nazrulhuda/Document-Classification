# Making a Document Classification to identify Document types using Textract and Fitz

Firstly we need to make a dataset. Using **create_dataset.py** we can extract text from the documents. For text documents we are using a 
python librarby called **fitz** and for scan/image docucments we are using **textract** (an OCT from aws). As textract is expensive, we only scan important pages using trimming, which is shown in the repository **Document-Trimming-Classification**

After we make the dataset we train it using **retraining.py**

Finally, we can test our classifier using a document with **loading.py**


