import os
import fitz
from s3_operations import S3Ops
s3 = S3Ops()
import time
from textract_functions import TextractCall
class MyError(Exception):
    pass

tc = TextractCall() 
def process_files(input_folder, output_folder):
   

    for filename in os.listdir(input_folder):
        print(filename,"******")
        input_file = os.path.join(input_folder, filename)
        print(input_file)
        output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")

        try:
            sync_flag = 0

            with fitz.open(input_file) as doc:
                if doc.page_count == 1:
                    sync_flag = 1
                text = ""
                for i, page in enumerate(doc):
                    try:
                                page_text = page.get_text()
                                if page_text:
                                    text += " p*" + str(i+1) + "*\n" + page_text + "\n"
                                else:
                                    raise MyError("TextNotExtracted")
                    except RuntimeError as e:
                                print("Error processing page in PDF file:", input_file)
                                print(e)
                                continue
                

                with open(output_file, "w") as f:
                    f.write(text)

                

        except MyError as _:
            text = None
            
            bucket="textractneoo"
            print(input_file)
            try:
                if sync_flag == 1:
                    text = tc.detect_text_from_pc_sync(input_file)
                if sync_flag == 0:
                    print("1")
                    text = tc.detect_text_async(input_file[1:],"textractneoo")
                if text in ('', None):
                    raise

                with open(output_file, "w") as f:
                    f.write(text)

                

            except Exception as e:
                print("Error processing file:", input_file)
                print(e)

                

# Usage
input_folder = '/home/shanto/a'

# Central output folder path to save the converted text files
output_folder = '/home/shanto/d'


process_files(input_folder, output_folder)
