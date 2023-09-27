import imp
import os 
import time
from minio import Minio
import logs
from utils import logs
import configparser
# from minio.error import ResponseError
from minio.error import S3Error


class minio_write(object):
    def __init__(self):
        self.logger = logs.Log("algorithm").logs_setup()
        self.logger.info("Minio client init")
        path = os.path.abspath(".")
        config_path = os.path.join(path, 'configs', 'config.ini')
        config = configparser.ConfigParser()
        config.read(config_path, encoding='utf-8')
        self.minio_host = config.get('Minio', "minio_host")
        self.minio_port = config.get('Minio', "minio_port")
        self.minio_access_key = config.get('Minio', "minio_access_key")
        self.minio_secret_key = config.get('Minio', "minio_secret_key")
        self.minio_bucket = config.get('Minio', "minio_bucket")
    
    def run(self, object_file, local_file):
        try:
            self.logger.info("Uploading file: %s to bucket: %s", object_file, self.minio_bucket)
            with open(local_file, 'rb') as file_data:
                file_stat = os.stat(local_file)
                minioClient = Minio(self.minio_host + ":" + self.minio_port,
                                    access_key=self.minio_access_key,
                                    secret_key=self.minio_secret_key,
                                    secure=False)
                minioClient.put_object(self.minio_bucket, object_file, file_data, file_stat.st_size)
        # except ResponseError as err:
        except S3Error as err:
            print(err)
        
