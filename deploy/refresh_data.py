import boto3
import requests
import os

s3 = boto3.client("s3")
bucket = os.environ["S3_BUCKET"]

# Example: delete all existing files in the bucket
def clear_bucket():
    response = s3.list_objects_v2(Bucket=bucket)
    if "Contents" in response:
        for obj in response["Contents"]:
            s3.delete_object(Bucket=bucket, Key=obj["Key"])
        print("Old files deleted.")

# Example: download new files and upload to S3
def upload_new_files():
    urls = [
        "https://example.com/file1.csv",
        "https://example.com/file2.csv"
    ]
    for url in urls:
        filename = url.split("/")[-1]
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)
        s3.upload_file(filename, bucket, filename)
        print(f"{filename} uploaded.")

if __name__ == "__main__":
    clear_bucket()
    upload_new_files()
