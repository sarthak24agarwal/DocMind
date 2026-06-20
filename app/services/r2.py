import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class R2Service:
    def __init__(self):
        # R2 requires endpoint_url to be specified, plus signature_version='s3v4'
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )
        self.bucket_name = settings.R2_BUCKET_NAME

    def upload_fileobj(self, file_obj, object_name: str) -> str:
        """
        Uploads a file-like object to R2.
        Returns the object name / key.
        """
        try:
            self.s3_client.upload_fileobj(file_obj, self.bucket_name, object_name)
            logger.info(f"Successfully uploaded {object_name} to R2")
            return object_name
        except ClientError as e:
            logger.error(f"Failed to upload {object_name} to R2: {str(e)}")
            raise e

    def download_file(self, object_name: str, local_path: str) -> None:
        """
        Downloads a file from R2 to a local path.
        """
        try:
            self.s3_client.download_file(self.bucket_name, object_name, local_path)
            logger.info(f"Successfully downloaded {object_name} from R2 to {local_path}")
        except ClientError as e:
            logger.error(f"Failed to download {object_name} from R2: {str(e)}")
            raise e

    def generate_presigned_url(self, object_name: str, expiration: int = 3600) -> str:
        """
        Generates a pre-signed URL to share/read the file directly.
        """
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_name},
                ExpiresIn=expiration,
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for {object_name}: {str(e)}")
            raise e

# Instantiate service singleton
r2_service = R2Service()
