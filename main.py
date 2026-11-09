"""
Main entry point for the Defect Detection Project.

Usage:
    # Full pipeline (data -> GAN -> YOLO -> evaluate -> export)
    python main.py --pipeline full

    # Quick start with default settings
    python main.py --pipeline quick

    # Individual steps
    python main.py --step download_data
    python main.py --step preprocess
    python main.py --step train_gan
    python main.py --step generate_gan
    python main.py --step train_yolo
    python main.py --step evaluate
    python main.py --step export
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.common import setup_logger, set_seed, create_directories


def run_step(step: str, config: dict, logger):
    """Run a single pipeline step."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Step: {step}")
    logger.info(f"{'='*60}")

    if step == "download_data":
        from scripts.download_data import download_neu_dataset, check_dataset_structure
        logger.info("Downloading NEU dataset...")
        success = download_neu_dataset(config['data_raw_dir'])
        if not success:
            logger.warning("Automatic download failed. Please download manually.")
            logger.info("Checking if dataset already exists...")
            check_dataset_structure(config['data_raw_dir'], "NEU")

    elif step == "preprocess":
        logger.info("Preprocessing data...")
        import subprocess
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "preprocess_data.py"),
            "--dataset", "NEU",
            "--raw_dir", config['data_raw_dir'],
            "--output_dir", config['yolo_data_dir'],
            "--gan_output_dir", config['gan_data_dir'],
            "--image_size", str(config['gan_image_size']),
            "--seed", str(config['seed']),
        ]
        subprocess.run(cmd, check=True)

    elif step == "train_gan":
        logger.info("Training GAN...")
        import subprocess
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "train_gan.py"),
            "--config", str(PROJECT_ROOT / "configs" / "gan_config.yaml"),
            "--model", config['gan_model'],
            "--epochs", str(config['gan_epochs']),
            "--batch_size", str(config['gan_batch_size']),
        ]
        subprocess.run(cmd, check=True)

    elif step == "generate_gan":
        logger.info("Generating GAN samples...")
        import subprocess
        checkpoint = PROJECT_ROOT / "outputs" / "checkpoints" / config['gan_model'] / "best_model.pth"
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "generate_samples.py"),
            "--checkpoint", str(checkpoint),
            "--model", config['gan_model'],
            "--num_per_class", str(config['gan_num_per_class']),
            "--output_dir", config['gan_generated_dir'],
        ]
        subprocess.run(cmd, check=True)

    elif step == "mix_data":
        logger.info("Mixing real and GAN data...")
        import subprocess
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "preprocess_data.py"),
            "--raw_dir", config['data_raw_dir'],
            "--output_dir", config['yolo_data_dir'],
            "--gan_dir", config['gan_generated_dir'],
            "--gan_ratio", str(config['gan_ratio']),
        ]
        subprocess.run(cmd, check=True)

    elif step == "train_yolo":
        logger.info("Training YOLO model...")
        import subprocess
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "train_yolo.py"),
            "--config", str(PROJECT_ROOT / "configs" / "yolo_config.yaml"),
            "--model", config['yolo_model'],
            "--epochs", str(config['yolo_epochs']),
            "--batch_size", str(config['yolo_batch_size']),
            "--imgsz", str(config['yolo_imgsz']),
        ]
        subprocess.run(cmd, check=True)

    elif step == "evaluate":
        logger.info("Evaluating model...")
        import subprocess
        weights = PROJECT_ROOT / "outputs" / "yolo_runs" / config['yolo_model'] / "weights" / "best.pt"
        data_yaml = Path(config['yolo_data_dir']) / "data.yaml"
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--weights", str(weights),
            "--data", str(data_yaml),
            "--output_dir", str(PROJECT_ROOT / "outputs" / "results" / "evaluation"),
            "--imgsz", str(config['yolo_imgsz']),
        ]
        subprocess.run(cmd, check=True)

    elif step == "export":
        logger.info("Exporting model...")
        import subprocess
        weights = PROJECT_ROOT / "outputs" / "yolo_runs" / config['yolo_model'] / "weights" / "best.pt"
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "export_model.py"),
            "--weights", str(weights),
            "--format", config['export_format'],
            "--imgsz", str(config['yolo_imgsz']),
            "--simplify",
            "--verify",
        ]
        subprocess.run(cmd, check=True)

    else:
        logger.error(f"Unknown step: {step}")
        return False

    logger.info(f"Step '{step}' completed successfully!")
    return True


def run_pipeline(pipeline_name: str, config: dict, logger):
    """Run a full pipeline."""
    pipelines = {
        "full": [
            "download_data",
            "preprocess",
            "train_gan",
            "generate_gan",
            "mix_data",
            "train_yolo",
            "evaluate",
            "export",
        ],
        "quick": [
            "preprocess",
            "train_yolo",
            "evaluate",
        ],
        "gan_only": [
            "preprocess",
            "train_gan",
            "generate_gan",
        ],
        "yolo_only": [
            "preprocess",
            "train_yolo",
            "evaluate",
            "export",
        ],
    }

    if pipeline_name not in pipelines:
        logger.error(f"Unknown pipeline: {pipeline_name}")
        logger.info(f"Available pipelines: {list(pipelines.keys())}")
        return False

    steps = pipelines[pipeline_name]
    logger.info(f"Running pipeline: {pipeline_name}")
    logger.info(f"Steps: {steps}")

    for i, step in enumerate(steps):
        logger.info(f"\nProgress: {i+1}/{len(steps)}")
        try:
            run_step(step, config, logger)
        except Exception as e:
            logger.error(f"Step '{step}' failed: {e}")
            logger.error("Pipeline aborted.")
            return False

    logger.info(f"\n{'='*60}")
    logger.info("Pipeline completed successfully!")
    logger.info(f"{'='*60}")
    return True


def get_default_config():
    """Get default configuration."""
    return {
        'seed': 42,
        'data_raw_dir': str(PROJECT_ROOT / "data" / "raw" / "NEU-CLS"),
        'yolo_data_dir': str(PROJECT_ROOT / "data" / "processed" / "yolo_format"),
        'gan_data_dir': str(PROJECT_ROOT / "data" / "processed" / "gan_dataset"),
        'gan_generated_dir': str(PROJECT_ROOT / "data" / "processed" / "gan_generated"),
        'gan_model': 'dcgan',
        'gan_epochs': 200,
        'gan_batch_size': 64,
        'gan_image_size': 64,
        'gan_num_per_class': 300,
        'gan_ratio': 0.5,
        'yolo_model': 'yolov8n',
        'yolo_epochs': 300,
        'yolo_batch_size': 16,
        'yolo_imgsz': 640,
        'export_format': 'onnx',
    }


def main():
    parser = argparse.ArgumentParser(
        description="GAN+YOLO Industrial Defect Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --pipeline full

  # Quick start (YOLO only, no GAN)
  python main.py --pipeline quick

  # Run individual step
  python main.py --step preprocess
  python main.py --step train_yolo
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pipeline", type=str,
                       choices=["full", "quick", "gan_only", "yolo_only"],
                       help="Run a predefined pipeline")
    group.add_argument("--step", type=str,
                       choices=["download_data", "preprocess", "train_gan",
                                "generate_gan", "mix_data", "train_yolo",
                                "evaluate", "export"],
                       help="Run a single step")

    # Model options
    parser.add_argument("--gan_model", type=str, default="dcgan",
                        choices=["dcgan", "cdgan"],
                        help="GAN model type")
    parser.add_argument("--yolo_model", type=str, default="yolov8n",
                        help="YOLO model name")
    parser.add_argument("--yolo_epochs", type=int, default=300,
                        help="YOLO training epochs")
    parser.add_argument("--gan_epochs", type=int, default=200,
                        help="GAN training epochs")

    args = parser.parse_args()

    # Setup
    config = get_default_config()
    config['gan_model'] = args.gan_model
    config['yolo_model'] = args.yolo_model
    config['yolo_epochs'] = args.yolo_epochs
    config['gan_epochs'] = args.gan_epochs

    set_seed(config['seed'])

    create_directories([
        PROJECT_ROOT / "outputs" / "logs",
    ])

    logger = setup_logger(str(PROJECT_ROOT / "outputs" / "logs"), "main")

    logger.info("=" * 60)
    logger.info("GAN+YOLO Industrial Defect Detection System")
    logger.info("第二届全国人工智能应用创新大赛 - 制造方向")
    logger.info("=" * 60)

    if args.pipeline:
        run_pipeline(args.pipeline, config, logger)
    elif args.step:
        run_step(args.step, config, logger)


if __name__ == "__main__":
    main()
