from pathlib import Path
from args import get_args

args = get_args()
folders = [
        Path(args.csv_dir),
        Path(args.output_dir),
        Path(args.model_dir),
        Path(args.visual_dir),
        Path(args.checkpoint_dir),
        Path(args.metrics_dir),
        Path("data"),
    ]
for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)