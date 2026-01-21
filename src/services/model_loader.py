import os

import onnxruntime as ort
from ultralytics import YOLO

def load_onnx_model():
    model_path = 'models/fridges-base-yolo9c-001-BEST/weights/best.onnx'
    if os.path.exists(model_path):
        return ort.InferenceSession(model_path)

def load_fridges_YOLO_model():
    model_path = 'models/fridges-base-yolo9c-001-BEST/weights/best.pt'
    if os.path.exists(model_path):
        return YOLO(model_path)
    
def load_shelves_YOLO_model():
    model_path = 'models/shelves-base-yolo9s-001-BEST/weights/best.pt'
    if os.path.exists(model_path):
        return YOLO(model_path)
