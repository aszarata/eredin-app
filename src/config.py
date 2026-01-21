import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

DB_PATH = 'database.db'
CONFIDENCE = 0.9

FRIDGES_MODEL_ONNX = 'models/fridges-base-yolo9c-001-BEST/weights/best.onnx'
FRIDGES_MODEL_PT   = 'models/fridges-base-yolo9c-001-BEST/weights/best.pt'
SHELVES_MODEL_PT   = 'models/shelves-base-yolo9s-001-BEST/weights/best.pt'

SHELVES_MAX_MARGIN = 2
SHELVES_MAX_ANGLE = 20