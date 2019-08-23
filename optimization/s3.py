import cPickle as pickle
import logging
import os

import boto3.session


class S3(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
        self.aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']

    def get_bucket_by_name(self, bucket_name):
        session = boto3.session.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )
        s3 = session.resource('s3')

        return s3.Bucket(bucket_name)

    def put_object(self, bucket_name, key, data):
        bucket = self.get_bucket_by_name(bucket_name)

        return bucket.put_object(Key=key, Body=pickle.dumps(data))

S3_handler = S3()
