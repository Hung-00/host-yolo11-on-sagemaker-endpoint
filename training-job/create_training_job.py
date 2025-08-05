import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from datetime import datetime


def lambda_handler(event, context):
    print(event)
    sagemaker_session = sagemaker.Session()
    model_data = f's3://uniben-data/output_lambda/{event.get("train_folder")}/output/model.tar.gz'
    model_name = f'yolo11x-model-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    endpoint_name = f'yolo11x-endpoint-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    region = boto3.Session().region_name
    pytorch_inference_image = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="2.0.0",
        py_version="py310",
        image_scope="inference",
        instance_type="ml.m5.xlarge",
    )

    model = Model(
        model_data=model_data,
        image_uri=pytorch_inference_image,
        role="arn:aws:iam:::role/YOLO11SageMakerStack",
        name=model_name,
        sagemaker_session=sagemaker_session,
        env={
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": model_data,
            "SAGEMAKER_REGION": region,
            "TS_MAX_RESPONSE_SIZE": "20000000",
            "YOLO11_MODEL": "model.pt",
        },
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=endpoint_name,
        # wait=True
    )

    print(f"Model deployed successfully!")
    print(f"Model name: {model_name}")
    print(f"Endpoint name: {endpoint_name}")

    return {
        "statusCode": 200,
        "body": {
            "model_data": model_data,
            "model_name": model_name,
            "endpoint_name": endpoint_name,
        },
    }
