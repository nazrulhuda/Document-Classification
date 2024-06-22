import os
import fitz
import time
from textract_functions import TextractCall

class MyError(Exception):
    pass


tc = TextractCall()


def process_files(input_folder, output_folder):
    for root, folders, files in os.walk(input_folder):
        for filename in files:
            input_file = os.path.join(root, filename)
            relative_folder = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, relative_folder)
            os.makedirs(output_subfolder, exist_ok=True)
            output_file = os.path.join(output_subfolder, os.path.splitext(filename)[0] + ".txt")

            try:
                sync_flag = 0
                if os.path.getsize(input_file) == 0:
                    print("Skipping empty file:", input_file)
                    continue
                if not filename.endswith(".pdf"):
                   continue
                
                   
                with fitz.open(input_file) as doc:
                        if doc.page_count == 1:
                            sync_flag = 1
                        text = ""
                        for i, page in enumerate(doc):
                            page_text = page.get_text()
                            if page_text:
                                text += " p*" + str(i+1) + "*\n" + page_text + "\n"
                            else:
                                raise MyError("TextNotExtracted")

                        with open(output_file, "w") as f:
                            f.write(text)
                

            except MyError as _:
                text = None
                text_extraction_start_time = time.time()
                try:
                    if sync_flag == 1:
                        text = tc.detect_text_from_pc_sync(input_file)
                    if sync_flag == 0:
                        text = tc.detect_text_async(input_file[1:], "textractneoo")
                    if text in ('', None):
                        raise

                    with open(output_file, "w") as f:
                        f.write(text)

                except Exception as e:
                    print("Error processing file:", input_file)
                    print(e)
                    continue


# Usage
input_folder = '/home/shanto/Downloads/sdsd/test/to/data'

# Central output folder path to save the converted text files
output_folder = '/home/shanto/Downloads/sdsd/test/to/dataset'

process_files(input_folder, output_folder)


