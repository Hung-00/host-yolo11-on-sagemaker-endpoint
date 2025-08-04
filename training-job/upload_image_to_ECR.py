# build_and_push.py
import boto3
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path


def get_account_id():
    """Get AWS account ID"""
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def get_region():
    """Get AWS region"""
    session = boto3.Session()
    return session.region_name


def get_timestamp_tag():
    """Generate timestamp tag in format YYYYMMDD-HHMMSS"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def create_ecr_repository(repository_name, region):
    """Create ECR repository if it doesn't exist"""
    ecr = boto3.client("ecr", region_name=region)

    try:
        ecr.describe_repositories(repositoryNames=[repository_name])
        print(f"Repository {repository_name} already exists")
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(repositoryName=repository_name)
        print(f"Created repository {repository_name}")


def build_and_push_docker_image():
    """Build and push Docker image to ECR with timestamp tag"""

    # Configuration
    account_id = get_account_id()
    region = get_region()
    repository_name = "yolo11-training"
    timestamp_tag = get_timestamp_tag()

    # Create both timestamp and latest tags
    timestamp_image_uri = (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}:{timestamp_tag}"
    )
    latest_image_uri = (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}:latest"
    )

    print(f"Account ID: {account_id}")
    print(f"Region: {region}")
    print(f"Repository: {repository_name}")
    print(f"Timestamp Tag: {timestamp_tag}")
    print(f"Timestamp Image URI: {timestamp_image_uri}")
    print(f"Latest Image URI: {latest_image_uri}")

    # Create ECR repository
    create_ecr_repository(repository_name, region)

    # Login to ECR
    print("Logging in to ECR...")
    login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    subprocess.run(login_cmd, shell=True, check=True)

    # Build Docker image
    print("Building Docker image...")
    build_cmd = f"docker build -t {repository_name} ."
    subprocess.run(build_cmd, shell=True, check=True)

    # Tag the image with timestamp
    print(f"Tagging image with timestamp: {timestamp_tag}")
    tag_timestamp_cmd = f"docker tag {repository_name}:latest {timestamp_image_uri}"
    subprocess.run(tag_timestamp_cmd, shell=True, check=True)

    # Tag the image with latest
    print("Tagging image with latest...")
    tag_latest_cmd = f"docker tag {repository_name}:latest {latest_image_uri}"
    subprocess.run(tag_latest_cmd, shell=True, check=True)

    # Push timestamp-tagged image to ECR
    print(f"Pushing timestamp-tagged image to ECR...")
    push_timestamp_cmd = f"docker push {timestamp_image_uri}"
    subprocess.run(push_timestamp_cmd, shell=True, check=True)

    # Push latest-tagged image to ECR
    print("Pushing latest-tagged image to ECR...")
    push_latest_cmd = f"docker push {latest_image_uri}"
    subprocess.run(push_latest_cmd, shell=True, check=True)

    print(f"\nDocker images pushed successfully!")
    print(f"Timestamp Image URI: {timestamp_image_uri}")
    print(f"Latest Image URI: {latest_image_uri}")

    return timestamp_image_uri, latest_image_uri


if __name__ == "__main__":
    # Ensure we're in the right directory
    if not os.path.exists("Dockerfile"):
        print("Error: Dockerfile not found in current directory")
        print("Please run this script from the directory containing the Dockerfile")
        sys.exit(1)

    timestamp_uri, latest_uri = build_and_push_docker_image()
