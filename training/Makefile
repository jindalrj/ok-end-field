# Tesseract Fine-Tuning for Endfield
#
# Prerequisites:
#   - tesseract installed with training tools (lstmtraining, combine_tessdata)
#   - Python 3.10+ with: pip install pytesseract pillow opencv-python numpy
#
# Usage:
#   make generate   - Generate training data (images + ground truth)
#   make train      - Run LSTM fine-tuning (takes 5-15 minutes)
#   make deploy     - Copy trained model to project tessdata
#   make all        - Full pipeline
#
# On Windows (no make): run the Python scripts directly:
#   python generate_training_data.py
#   python annotate_tooltips.py
#   python run_training.py

MODEL_NAME = endfield
START_MODEL = eng
MAX_ITERATIONS = 3000

GROUND_TRUTH_DIR = tesstrain/ground-truth
OUTPUT_DIR = tesstrain/output
LSTMF_DIR = tesstrain/lstmf

.PHONY: all generate annotate train deploy clean

all: generate annotate train

generate:
	python generate_training_data.py

annotate:
	python annotate_tooltips.py

train:
	python run_training.py

deploy:
	@echo "Copying $(OUTPUT_DIR)/$(MODEL_NAME).traineddata to tessdata..."
	@if [ -f "$(OUTPUT_DIR)/$(MODEL_NAME).traineddata" ]; then \
		cp "$(OUTPUT_DIR)/$(MODEL_NAME).traineddata" ../runtime/tesseract/tessdata/ 2>/dev/null || \
		echo "Copy failed - manually copy to your tessdata directory"; \
	else \
		echo "Trained model not found. Run 'make train' first."; \
	fi

clean:
	rm -rf tesstrain/ground-truth tesstrain/lstmf tesstrain/output
