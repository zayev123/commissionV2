from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
from storages.backends.s3boto3 import S3Boto3Storage
import logging
import re
from rest_framework import serializers
from commissionerv2.settings import MEDIA_ROOT

logger = logging.getLogger(__name__)
class MediaStorage(S3Boto3Storage):
    bucket_name = os.environ.get("BUCKETNAME")
    location = 'media'
    file_overwrite = True

class StaticStorage(S3Boto3Storage):
    bucket_name = os.environ.get("BUCKETNAME")
    location = 'static'

def __get_storage_class():
    if os.environ.get("OS_ENV") == "PROD":
        media_storage_class = MediaStorage
    else:
        media_storage_class = FileSystemStorage
    return media_storage_class

class OverwriteStorage(__get_storage_class()):
    def get_available_name(self, name, max_length=None):
        if os.environ.get("OS_ENV") != "PROD":
            if self.exists(name):
                os_temp_path = os.path.join(MEDIA_ROOT, name)
                os.remove(os_temp_path)
        return name