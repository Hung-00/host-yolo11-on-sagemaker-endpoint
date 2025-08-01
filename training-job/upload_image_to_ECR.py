# build_and_push.py
import boto3
import subprocess
import sys
import os
from pathlib import Path


def get_account_id():
    """Get AWS account ID"""
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def get_region():
    """Get AWS region"""
    session = boto3.Session()
    return session.region_name


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
    """Build and push Docker image to ECR"""

    # Configuration
    account_id = get_account_id()
    region = get_region()
    repository_name = "yolo11-training"
    image_tag = "latest"

    # Full image URI
    image_uri = (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}:{image_tag}"
    )

    print(f"Account ID: {account_id}")
    print(f"Region: {region}")
    print(f"Repository: {repository_name}")
    print(f"Image URI: {image_uri}")

    # Create ECR repository
    create_ecr_repository(repository_name, region)

    # Get ECR login token
    print("Getting ECR login token...")
    ecr = boto3.client("ecr", region_name=region)
    auth_token = ecr.get_authorization_token()["authorizationData"][0][
        "authorizationToken"
    ]

    # Login to ECR
    print("Logging in to ECR...")
    login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    subprocess.run(login_cmd, shell=True, check=True)

    # Build Docker image
    print("Building Docker image...")
    build_cmd = f"docker build -t {repository_name} ."
    subprocess.run(build_cmd, shell=True, check=True)

    # Tag the image
    print("Tagging image...")
    tag_cmd = f"docker tag {repository_name}:latest {image_uri}"
    subprocess.run(tag_cmd, shell=True, check=True)

    # Push to ECR
    print("Pushing image to ECR...")
    push_cmd = f"docker push {image_uri}"
    subprocess.run(push_cmd, shell=True, check=True)

    print(f"\nDocker image pushed successfully!")
    print(f"Image URI: {image_uri}")

    return image_uri


if __name__ == "__main__":
    # Ensure we're in the right directory
    if not os.path.exists("Dockerfile"):
        print("Error: Dockerfile not found in current directory")
        print("Please run this script from the directory containing the Dockerfile")
        sys.exit(1)

    image_uri = build_and_push_docker_image()
