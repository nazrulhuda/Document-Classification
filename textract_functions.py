"""
Code to define textract functions and parse the respose
"""
import time
import logging
import boto3
from botocore.client import Config as boto_config
from Config import Config
from sql_operations import SQL
import json
print("dadsdsd")

logger = logging.getLogger(__name__)
logger.propagate = False
print("tx2")
sql = SQL()
print("tx3")


class TextractCall:
    """
    A class to define Textract methods.
    """

    def __init__(self):

        """
        A constructor to intilize AWS client
        """

        config = boto_config(retries = dict(max_attempts = Config.max_attempts))
        self.client = boto3.client('textract',
                                    config=config,
                                    aws_access_key_id = Config.aws_access_key_id,
                                    aws_secret_access_key = Config.aws_secret_access_key,
                                    region_name=Config.region_name)


        
        

    def start_text_job(self, document_name, bucket):
        """
        A Textract method to start document analysis on provided document.

        Parameters
        ----------
        document_name : str
            The name of the document in S3 bucket.
        bucket : str
            name of bucket in s3.

        Returns
        -------
        JobId: str
            Unique identiier assigned by aws for our job.

        """

        response = None
        response = self.client.start_document_text_detection(
            DocumentLocation={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': document_name
                }})

        return response["JobId"]


    def is_text_job_complete(self, job_id):
        """
        Accepts jobid assigned by AWS during start of job and checks if
        the job is in completed state.

        Parameters
        ----------
        job_id : str
            Unique identiier assigned by aws for our job.

        Returns
        -------
        status : str
            Can be one of 'IN_PROGRESS'|'SUCCEEDED'|'FAILED'|'PARTIAL_SUCCESS'.

        """

        time.sleep(1)
        response = self.client.get_document_text_detection(JobId=job_id)
        status = response["JobStatus"]

        while status == "IN_PROGRESS":
            time.sleep(1)
            response = self.client.get_document_text_detection(JobId=job_id)
            status = response["JobStatus"]

        return status


    def get_text_job_results(self, job_id):
        """
        Accepts jobid assigned by AWS during start of job and checks if
        the job is in completed state and fetches results.

        Parameters
        ----------
        job_id : str
            Unique identiier assigned by aws for our job..

        Returns
        -------
        pages : list
            result/response of each page in the document.

        """

        pages = []
        time.sleep(1)
        response = self.client.get_document_text_detection(JobId=job_id)
        pages.append(response)
        next_token = None
        if 'NextToken' in response:
            next_token = response['NextToken']

        while next_token:
            time.sleep(1)
            response = self.client.\
                get_document_text_detection(JobId=job_id, NextToken=next_token)
            pages.append(response)
            next_token = None
            if 'NextToken' in response:
                next_token = response['NextToken']

        return pages
   

    def detect_text_async(self, document_name, bucket):
        """
        A method to extract plain text from document

        Parameters
        ----------
        document_name : str
            The name of the document in S3 bucket.
        bucket : str
            name of bucket in s3.

        Returns
        -------
        None.

        """

        job_id = self.start_text_job(document_name, bucket)
        if self.is_text_job_complete(job_id):
            response = self.get_text_job_results(job_id)

        text = ""
        for result_page in response:
            for item in result_page["Blocks"]:
                if item["BlockType"] == "LINE":
                    text = text + item["Text"] + "\n"
        return text
    
    ##########################################
    ##########################################
    #Using textract to extract from local 
    
    def detect_text_from_pc_sync(self, file_path):
        with open(file_path, 'rb') as f:
          file_bytes = f.read()
        response = None
        logger.debug('Started extracting only text from document')
        response = self.client.detect_document_text(Document={'Bytes': file_bytes})
        text = ""
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                text = text + item["Text"] + '\n'
        return text
    
    
   
        

    def detect_text_sync(self, document_name, bucket):
        """
        A method to extract plain text from document

        Parameters
        ----------
        document_name : str
            The name of the document in S3 bucket.
        bucket : str
            name of bucket in s3.

        Returns
        -------
        None.

        """

        response = None
        logger.debug('Started extracting only text from document: %s', document_name)
        response = self.client.detect_document_text(
                   Document={ 'S3Object': {'Bucket': bucket,
                                           'Name': document_name}})
        text = ""
        for item in response["Blocks"]:
            if item["BlockType"] == "LINE":
                text = text + item["Text"] + '\n'
        return text

