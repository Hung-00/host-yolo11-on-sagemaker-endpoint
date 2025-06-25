#!/usr/bin/env python3
import os
import aws_cdk as cdk
from cdk.yolo11_sagemaker import YOLO11SageMakerStack

app = cdk.App()
YOLO11SageMakerStack(
    app,
    "YOLO11SageMakerStack",
)

app.synth()
