"""
Run Tesseract LSTM fine-tuning on generated training data.

This script:
1. Checks prerequisites (tesseract, training tools)
2. Generates lstmf files from ground-truth pairs
3. Runs lstmtraining to fine-tune from eng base model
4. Outputs endfield.traineddata ready for deployment

Usage:
    python training/run_training.py

The output traineddata is saved to:
    training/tesstrain/output/endfield.traineddata

To deploy, copy it to the tessdata directory alongside the Tesseract binary.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training" / "tesstrain"
GT_DIR = TRAINING_DIR / "ground-truth"
OUTPUT_DIR = TRAINING_DIR / "output"
LSTMF_DIR = TRAINING_DIR / "lstmf"

MODEL_NAME = "endfield"
START_MODEL = "eng"  # fine-tune from English base
MAX_ITERATIONS = 3000


def find_tesseract() -> str:
    """Find tesseract binary."""
    # Check project's bundled tesseract first
    bundled = PROJECT_ROOT / "runtime" / "tesseract" / "tesseract.exe"
    if bundled.exists():
        return str(bundled)
    # Check PATH
    import shutil
    t = shutil.which("tesseract")
    if t:
        return t
    raise FileNotFoundError("tesseract not found. Install or set PATH.")


def find_training_tools() -> dict:
    """Find tesstrain tools (lstmtraining, combine_tessdata, etc.)."""
    import shutil
    tools = {}
    for name in ["lstmtraining", "combine_tessdata", "tesseract"]:
        path = shutil.which(name)
        if path:
            tools[name] = path
    return tools


def get_tessdata_dir(tesseract_path: str) -> Path:
    """Get tessdata directory from tesseract installation."""
    tess_dir = Path(tesseract_path).parent
    tessdata = tess_dir / "tessdata"
    if tessdata.exists():
        return tessdata
    # Try share/tessdata (Linux)
    share = tess_dir.parent / "share" / "tessdata"
    if share.exists():
        return share
    # Try TESSDATA_PREFIX env
    prefix = os.environ.get("TESSDATA_PREFIX")
    if prefix and Path(prefix).exists():
        return Path(prefix)
    raise FileNotFoundError(f"tessdata not found near {tesseract_path}")


def generate_lstmf_files(tesseract_path: str):
    """Generate .lstmf files from ground-truth .tif + .gt.txt pairs."""
    LSTMF_DIR.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(GT_DIR.glob("*.gt.txt"))
    if not gt_files:
        raise FileNotFoundError(f"No ground truth files in {GT_DIR}")

    # Filter out unannotated files
    valid_files = []
    for gt_file in gt_files:
        text = gt_file.read_text(encoding="utf-8").strip()
        if text == "__NEEDS_ANNOTATION__":
            continue
        tif_file = gt_file.with_suffix("").with_suffix(".tif")
        if tif_file.exists():
            valid_files.append((tif_file, gt_file))

    if not valid_files:
        raise ValueError("No annotated training data found. Run generate_training_data.py "
                         "and annotate the .gt.txt files first.")

    print(f"Generating lstmf from {len(valid_files)} annotated pairs...")

    for tif_file, gt_file in valid_files:
        lstmf_file = LSTMF_DIR / f"{tif_file.stem}.lstmf"
        if lstmf_file.exists():
            continue
        # tesseract <image> <output_base> --psm 7 lstm.train
        cmd = [
            tesseract_path, str(tif_file),
            str(LSTMF_DIR / tif_file.stem),
            "--psm", "7", "lstm.train"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARN: failed for {tif_file.name}: {result.stderr[:100]}")

    lstmf_files = list(LSTMF_DIR.glob("*.lstmf"))
    print(f"Generated {len(lstmf_files)} lstmf files")
    return lstmf_files


def create_training_list(lstmf_files: list[Path]) -> Path:
    """Create the training file list."""
    list_file = TRAINING_DIR / "training_files.txt"
    list_file.write_text("\n".join(str(f) for f in lstmf_files), encoding="utf-8")
    return list_file


def run_lstmtraining(tools: dict, tessdata_dir: Path, training_list: Path):
    """Run lstmtraining to fine-tune the model."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lstmtraining = tools.get("lstmtraining")
    if not lstmtraining:
        # Try in same dir as tesseract
        tess_dir = Path(tools["tesseract"]).parent
        candidate = tess_dir / "lstmtraining"
        if candidate.exists():
            lstmtraining = str(candidate)
        else:
            candidate = tess_dir / "lstmtraining.exe"
            if candidate.exists():
                lstmtraining = str(candidate)
            else:
                raise FileNotFoundError(
                    "lstmtraining not found. Install tesseract training tools:\n"
                    "  brew install tesseract --with-training-tools  (macOS)\n"
                    "  apt install tesseract-ocr libtesseract-dev  (Linux)\n"
                    "  Or download from: https://github.com/tesseract-ocr/tesseract"
                )

    # Extract starter model
    start_model = tessdata_dir / f"{START_MODEL}.traineddata"
    if not start_model.exists():
        raise FileNotFoundError(f"Base model not found: {start_model}")

    # combine_tessdata -e to extract LSTM model
    combine = tools.get("combine_tessdata", str(Path(lstmtraining).parent / "combine_tessdata"))
    lstm_file = TRAINING_DIR / f"{START_MODEL}.lstm"
    if not lstm_file.exists():
        cmd = [combine, "-e", str(start_model), str(lstm_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: combine_tessdata failed: {result.stderr}")
            # Try alternative extraction
            lstm_file = start_model  # Use full traineddata as starting point

    print(f"Starting fine-tuning from {START_MODEL} ({start_model})...")
    print(f"Max iterations: {MAX_ITERATIONS}")

    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        lstmtraining,
        "--model_output", str(checkpoint_dir / MODEL_NAME),
        "--continue_from", str(lstm_file),
        "--traineddata", str(start_model),
        "--train_listfile", str(training_list),
        "--max_iterations", str(MAX_ITERATIONS),
        "--target_error_rate", "0.01",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:] if result.stdout else "")
    if result.returncode != 0:
        print(f"ERROR: {result.stderr[-1000:]}")
        return False

    # Combine checkpoint into final traineddata
    final_checkpoint = str(checkpoint_dir / f"{MODEL_NAME}_checkpoint")
    final_output = OUTPUT_DIR / f"{MODEL_NAME}.traineddata"

    cmd = [
        lstmtraining,
        "--stop_training",
        "--continue_from", final_checkpoint,
        "--traineddata", str(start_model),
        "--model_output", str(final_output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR finalizing: {result.stderr}")
        return False

    print(f"\nTraining complete! Output: {final_output}")
    print(f"Size: {final_output.stat().st_size / 1024:.0f} KB")
    return True


def main():
    print("=" * 60)
    print("Tesseract Fine-Tuning for Endfield Item Names")
    print("=" * 60)
    print()

    tesseract_path = find_tesseract()
    print(f"Tesseract: {tesseract_path}")

    tools = find_training_tools()
    tools["tesseract"] = tesseract_path
    print(f"Tools found: {list(tools.keys())}")

    tessdata_dir = get_tessdata_dir(tesseract_path)
    print(f"Tessdata: {tessdata_dir}")
    print()

    # Step 1: Generate training data if not done
    if not list(GT_DIR.glob("*.tif")):
        print("No training data found. Run generate_training_data.py first:")
        print(f"  python {PROJECT_ROOT}/training/generate_training_data.py")
        return 1

    # Step 2: Generate lstmf files
    lstmf_files = generate_lstmf_files(tesseract_path)
    if not lstmf_files:
        return 1

    # Step 3: Create training list
    training_list = create_training_list(lstmf_files)

    # Step 4: Run training
    success = run_lstmtraining(tools, tessdata_dir, training_list)
    if not success:
        return 1

    print()
    print("To deploy the trained model:")
    print(f"  cp {OUTPUT_DIR / f'{MODEL_NAME}.traineddata'} <tessdata_dir>/")
    print("  Then set lang='endfield' in ocr_text() calls, or use eng+endfield")
    return 0


if __name__ == "__main__":
    sys.exit(main())
