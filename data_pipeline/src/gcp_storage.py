"""
Google Cloud Storage module.
Handles operations related to GCP Cloud Storage for the data pipeline.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor

from google.cloud import storage
from google.cloud.exceptions import NotFound
import dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Set default bucket
DEFAULT_BUCKET = os.getenv("GCP_BUCKET_NAME", "distinguish-ai-faces-dataset")


def setup_storage_client() -> storage.Client:
    """
    Set up GCP storage client with credentials.
    
    Returns:
        Google Cloud Storage client
    """
    # Priority 1: Explicitly set credential path in .env
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if creds_path and os.path.exists(creds_path):
        return storage.Client.from_service_account_json(creds_path)
    
    # Priority 2: Use default credentials (environment variable or default locations)
    try:
        return storage.Client()
    except Exception as e:
        logger.error(f"Failed to initialize GCP storage client: {str(e)}")
        raise


def validate_bucket_access(bucket_name: str = DEFAULT_BUCKET) -> bool:
    """
    Validate if the bucket is accessible.
    
    Args:
        bucket_name: Name of the GCS bucket
        
    Returns:
        Boolean indicating if bucket is accessible
    """
    try:
        client = setup_storage_client()
        bucket = client.bucket(bucket_name)
        return bucket.exists()
    except Exception as e:
        logger.error(f"Bucket access validation failed: {str(e)}")
        return False


def upload_file(
    local_file: Union[str, Path],
    bucket_name: str = DEFAULT_BUCKET,
    destination_path: Optional[str] = None,
    make_public: bool = False
) -> Dict[str, Any]:
    """
    Upload a file to GCS bucket.
    
    Args:
        local_file: Path to local file
        bucket_name: Name of the GCS bucket
        destination_path: Path within bucket (if None, use filename only)
        make_public: Whether to make the file publicly accessible
    
    Returns:
        Dictionary with upload status and details
    """
    try:
        local_file = Path(local_file)
        if not local_file.exists():
            return {"success": False, "error": f"File not found: {local_file}"}
        
        client = setup_storage_client()
        bucket = client.bucket(bucket_name)
        
        # Set destination blob name (path within bucket)
        if destination_path is None:
            destination_path = local_file.name
            
        blob = bucket.blob(destination_path)
        
        # Upload file
        blob.upload_from_filename(str(local_file))
        
        # Make public if requested
        if make_public:
            blob.make_public()
            
        return {
            "success": True,
            "bucket": bucket_name,
            "path": destination_path,
            "size": local_file.stat().st_size,
            "url": blob.public_url if make_public else None
        }
    
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        return {"success": False, "error": str(e)}


def upload_files_batch(
    local_files: List[Union[str, Path]],
    bucket_name: str = DEFAULT_BUCKET,
    destination_folder: str = "",
    make_public: bool = False,
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    Upload multiple files to GCS bucket in parallel.
    
    Args:
        local_files: List of paths to local files
        bucket_name: Name of the GCS bucket
        destination_folder: Folder within bucket to upload files to
        make_public: Whether to make the files publicly accessible
        max_workers: Maximum number of parallel uploads
    
    Returns:
        List of dictionaries with upload statuses
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_path in local_files:
            path = Path(file_path)
            dest_path = f"{destination_folder}/{path.name}" if destination_folder else path.name
            
            future = executor.submit(
                upload_file,
                local_file=file_path,
                bucket_name=bucket_name,
                destination_path=dest_path,
                make_public=make_public
            )
            futures.append(future)
        
        for future in futures:
            results.append(future.result())
    
    # Summary
    success_count = sum(1 for r in results if r.get("success", False))
    logger.info(f"Uploaded {success_count}/{len(results)} files to {bucket_name}")
    
    return results


def download_file(
    cloud_path: str,
    local_destination: Union[str, Path],
    bucket_name: str = DEFAULT_BUCKET
) -> Dict[str, Any]:
    """
    Download a file from GCS bucket.
    
    Args:
        cloud_path: Path within bucket to the file
        local_destination: Local path to save the file
        bucket_name: Name of the GCS bucket
    
    Returns:
        Dictionary with download status and details
    """
    try:
        client = setup_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(cloud_path)
        
        # Check if blob exists
        if not blob.exists():
            return {"success": False, "error": f"File not found in bucket: {cloud_path}"}
        
        # Ensure directory exists
        local_destination = Path(local_destination)
        local_destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file
        blob.download_to_filename(str(local_destination))
        
        return {
            "success": True,
            "bucket": bucket_name,
            "cloud_path": cloud_path,
            "local_path": str(local_destination),
            "size": local_destination.stat().st_size
        }
    
    except Exception as e:
        logger.error(f"File download failed: {str(e)}")
        return {"success": False, "error": str(e)}


def list_bucket_files(
    bucket_name: str = DEFAULT_BUCKET,
    prefix: str = "",
    delimiter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List files in a GCS bucket with optional filtering.
    
    Args:
        bucket_name: Name of the GCS bucket
        prefix: Filter results to objects with this prefix
        delimiter: Filter results to objects with this delimiter
    
    Returns:
        List of dictionaries with file details
    """
    try:
        client = setup_storage_client()
        bucket = client.bucket(bucket_name)
        
        # List objects
        blobs = client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
        
        # Convert to list of dictionaries
        file_list = []
        for blob in blobs:
            file_list.append({
                "name": blob.name,
                "size": blob.size,
                "updated": blob.updated,
                "content_type": blob.content_type,
                "url": blob.public_url
            })
        
        return file_list
    
    except Exception as e:
        logger.error(f"Failed to list bucket files: {str(e)}")
        return []


def delete_file(
    cloud_path: str,
    bucket_name: str = DEFAULT_BUCKET
) -> Dict[str, Any]:
    """
    Delete a file from GCS bucket.
    
    Args:
        cloud_path: Path within bucket to the file
        bucket_name: Name of the GCS bucket
    
    Returns:
        Dictionary with deletion status and details
    """
    try:
        client = setup_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(cloud_path)
        
        # Check if blob exists
        if not blob.exists():
            return {"success": False, "error": f"File not found in bucket: {cloud_path}"}
        
        # Delete file
        blob.delete()
        
        return {
            "success": True,
            "bucket": bucket_name,
            "deleted_path": cloud_path
        }
    
    except Exception as e:
        logger.error(f"File deletion failed: {str(e)}")
        return {"success": False, "error": str(e)} 