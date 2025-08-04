import os
import json
import logging
import sys
import torch
from ultralytics import YOLO
import argparse
import shutil
from pathlib import Path
import yaml
import traceback
import boto3
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
    parser.add_argument(
        "--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument(
        "--validation",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"
        ),
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )

    # YOLO training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--device", type=str, default="0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--project", type=str, default="/opt/ml/output")
    parser.add_argument("--name", type=str, default="yolo11x")
    parser.add_argument("--exist-ok", type=bool, default=True)
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False)

    return parser.parse_args()


def detect_dataset_structure(base_path):
    """Detect the structure of the dataset"""
    logger.info(f"Detecting dataset structure in: {base_path}")

    # List all files and directories
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, "").count(os.sep)
        indent = " " * 2 * level
        logger.info(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files[:10]:  # Show first 10 files
            logger.info(f"{subindent}{file}")
        if len(files) > 10:
            logger.info(f"{subindent}... and {len(files) - 10} more files")


def prepare_dataset(train_path, val_path):
    """Prepare dataset structure for YOLO training"""
    try:
        logger.info(f"Preparing dataset...")
        logger.info(f"Train path: {train_path}")
        logger.info(f"Validation path: {val_path}")

        # Check if paths exist
        if not os.path.exists(train_path):
            raise ValueError(f"Training path does not exist: {train_path}")
        if not os.path.exists(val_path):
            raise ValueError(f"Validation path does not exist: {val_path}")

        # Detect structure
        detect_dataset_structure(train_path)

        # Check for existing dataset.yaml in various locations
        possible_yaml_locations = [
            os.path.join(train_path, "dataset.yaml"),
            os.path.join(train_path, "../dataset.yaml"),
            # os.path.join("/opt/ml/input/data", "dataset.yaml"),
            # os.path.join("/opt/ml/input/data/train", "dataset.yaml"),
        ]

        dataset_yaml_path = None
        for yaml_path in possible_yaml_locations:
            if os.path.exists(yaml_path):
                logger.info(f"Found existing dataset.yaml at: {yaml_path}")
                dataset_yaml_path = yaml_path
                break

        # !!!!!!!!!!!!!!!!!!!!!!!
        if dataset_yaml_path:
            # Load and log existing configuration
            with open(dataset_yaml_path, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Existing dataset config: {json.dumps(config, indent=2)}")
            return dataset_yaml_path

        # Create dataset.yaml if it doesn't exist
        logger.info("Creating new dataset.yaml")

        # Try to auto-detect classes from labels
        classes = []
        labels_dir = os.path.join(train_path, "labels")
        if os.path.exists(labels_dir):
            # Read a few label files to detect number of classes
            label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
            max_class_id = -1
            for label_file in label_files[:10]:  # Check first 10 files
                with open(os.path.join(labels_dir, label_file), "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            max_class_id = max(max_class_id, class_id)

            # Create generic class names
            if max_class_id >= 0:
                classes = [f"class_{i}" for i in range(max_class_id + 1)]
                logger.info(f"Auto-detected {len(classes)} classes")

        # If no classes detected, use a default
        if not classes:
            logger.warning("Could not auto-detect classes, using default single class")
            classes = ["object"]

        # Create dataset configuration
        dataset_config = {
            "path": "/opt/ml/input/data",
            "train": "train",
            "val": "validation",
            "nc": len(classes),  # number of classes
            "names": classes,
        }

        # Save dataset.yaml
        output_yaml_path = "/opt/ml/input/data/dataset.yaml"
        os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)

        with open(output_yaml_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        logger.info(f"Created dataset.yaml at: {output_yaml_path}")
        logger.info(f"Dataset config: {json.dumps(dataset_config, indent=2)}")

        return output_yaml_path

    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def train():
    try:
        args = parse_args()

        logger.info("=" * 50)
        logger.info("Starting YOLO11x training")
        logger.info("=" * 50)
        logger.info(f"Arguments: {args}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"PyTorch version: {torch.__version__}")

        # Log environment variables
        logger.info("SageMaker environment variables:")
        for key, value in os.environ.items():
            if key.startswith("SM_"):
                logger.info(f"  {key}: {value}")

        # Create output directories
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.output_data_dir, exist_ok=True)
        os.makedirs(args.project, exist_ok=True)

        # Prepare dataset
        dataset_yaml = prepare_dataset(args.train, args.validation)

        # Initialize YOLO model
        logger.info("Initializing YOLO model...")
        if args.pretrained:
            logger.info("Using pretrained weights: yolo11x.pt")
            model = YOLO("yolo11x.pt")  # Will download pretrained weights
        else:
            logger.info("Training from scratch using: yolo11x.yaml")
            model = YOLO("yolo11x.yaml")

        logger.info("Model initialized successfully")

        # Train the model
        logger.info("Starting training...")
        results = model.train(
            data=dataset_yaml,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            lr0=args.learning_rate,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            resume=args.resume,
            verbose=True,
        )

        logger.info("Training completed!")

        # Save the best model to the model directory
        best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
        last_model_path = Path(args.project) / args.name / "weights" / "last.pt"

        # Try to save best model, fall back to last if best doesn't exist
        if best_model_path.exists():
            shutil.copy(str(best_model_path), os.path.join(args.model_dir, "model.pt"))
            logger.info(f"Best model saved to {args.model_dir}/model.pt")
        elif last_model_path.exists():
            shutil.copy(str(last_model_path), os.path.join(args.model_dir, "model.pt"))
            logger.info(f"Last model saved to {args.model_dir}/model.pt")
        else:
            logger.error("No model weights found to save!")

        # Copy code to model
        os.makedirs(os.path.join(args.model_dir, "code"), exist_ok=True)

        code_infer_path = Path("/opt/ml/code") / "inference.py"
        code_requirements_path = Path("/opt/ml/code") / "requirements.txt"
        shutil.copy(
            str(code_infer_path), os.path.join(args.model_dir, "code/inference.py")
        )
        shutil.copy(
            str(code_requirements_path),
            os.path.join(args.model_dir, "code/requirements.txt"),
        )

        # Upload results to S3
        results_dir = Path(args.project) / args.name
        if results_dir.exists():
            logger.info("Uploading training results to S3...")

            result_files = [
                "results.png",
                "confusion_matrix.png",
                "confusion_matrix_normalized.png",
                "BoxF1_curve.png",
                "BoxP_curve.png",
                "BoxR_curve.png",
                "BoxPR_curve.png",
                "labels.jpg",
                "labels_correlogram.jpg",
                "results.csv",
            ]

            for file in result_files:
                src_path = results_dir / file
                if src_path.exists():
                    dst_path = os.path.join(args.output_data_dir, file)
                    shutil.copy(str(src_path), dst_path)
                    logger.info(f"Copied {file} to output directory")

        # Save training metrics
        metrics = {
            "final_epoch": args.epochs,
            "training_completed": True,
        }

        with open(os.path.join(args.output_data_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)

        logger.info("Training job completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    train()
