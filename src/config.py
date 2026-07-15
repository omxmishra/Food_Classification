from pathlib import Path

DATA_DIR = Path("/home/om_mishra/food101_data/data")
MODELS_DIR = Path("models")

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

EPOCHS = 10
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LAYERS = 30

INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5