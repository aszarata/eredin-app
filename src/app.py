import os
import io
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from ultralytics import YOLO
from flask import Flask, request, render_template, jsonify, send_file
import sqlite3
import base64

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(BASE_DIR)              
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

CONFIDENCE = 0.1

DB_PATH = 'database.db'
onnx_session = None
yolo_model = None

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (id TEXT PRIMARY KEY, original_path TEXT, created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS bounding_boxes
                 (id TEXT PRIMARY KEY, image_id TEXT, bbox_path TEXT, 
                  x INTEGER, y INTEGER, w INTEGER, h INTEGER, created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_image_to_db(image_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    image_id = str(uuid.uuid4())
    c.execute("INSERT INTO images (id, original_path, created_at) VALUES (?, ?, ?)",
              (image_id, image_path, datetime.now()))
    conn.commit()
    conn.close()
    return image_id

def save_bbox_to_db(image_id, bbox_path, x, y, w, h):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    bbox_id = str(uuid.uuid4())
    c.execute('''INSERT INTO bounding_boxes 
                 (id, image_id, bbox_path, x, y, w, h, created_at) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (bbox_id, image_id, bbox_path, x, y, w, h, datetime.now()))
    conn.commit()
    conn.close()
    return bbox_id

def load_onnx_model():
    global onnx_session
    model_path = 'models/fridges-base-yolo9c-001-BEST/weights/best.onnx'
    if os.path.exists(model_path):
        onnx_session = ort.InferenceSession(model_path)

def load_YOLO_model():
    global yolo_model
    model_path = 'models/fridges-base-yolo9c-001-BEST/weights/best.pt'
    if os.path.exists(model_path):
        yolo_model = YOLO(model_path)

def run_model(image_path):
    # if not onnx_session:
    #     return None
    
    # img = Image.open(image_path).convert('RGB')
    # img_resized = img.resize((640, 640))
    # img_array = np.array(img_resized).astype(np.float32) / 255.0
    # img_array = img_array.transpose(2, 0, 1)  
    # img_array = img_array[np.newaxis, ...]

    # input_name = onnx_session.get_inputs()[0].name
    # outputs = onnx_session.run(None, {input_name: img_array})
    if not yolo_model:
        return None
    
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    outputs = yolo_model(img_array, conf=0.9)

    return process_output(outputs, img)

def process_output(outputs, original_img):
    if not outputs or len(outputs) == 0:
        return []
    
    boxes = []
    output_data = outputs[0].boxes.xyxy.cpu().numpy()
    for detection in output_data:
        x1, y1, x2, y2 = detection[:4]
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    # for detection in output_data:
    #     x_center = detection[0]
    #     y_center = detection[1]
    #     width = detection[2]
    #     height = detection[3]
        
    #     x1 = int((x_center - width/2))
    #     y1 = int((y_center - height/2))
    #     x2 = int((x_center + width/2))
    #     y2 = int((y_center + height/2))
            
    #     boxes.append((x1, y1, x2, y2))
    return boxes

def draw_boxes(image_path, boxes):
    img = cv2.imread(image_path)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    output_path = f"{UPLOAD_FOLDER}/bbox_{uuid.uuid4()}.jpg"
    cv2.imwrite(output_path, img)
    return output_path

def crop_and_save_bboxes(image_path, boxes):
    img = cv2.imread(image_path)
    cropped_paths = []
    print("AAAAA", img)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cropped = img[y1:y2, x1:x2]
        if cropped.size > 0:
            cropped_path = f"{UPLOAD_FOLDER}/cropped_{uuid.uuid4()}.jpg"
            cv2.imwrite(cropped_path, cropped)
            cropped_paths.append((cropped_path, x1, y1, x2-x1, y2-y1))
    
    return cropped_paths

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False})
    
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    image_id = save_image_to_db(filepath)
    
    return jsonify({
        'success': True,
        'image_id': image_id,
        'image_url': f'/image/{filename}'
    })

@app.route('/approve/<image_id>', methods=['POST'])
def approve_image(image_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT original_path FROM images WHERE id = ?", (image_id,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return jsonify({'success': False})
    
    filename = result[0]
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    boxes = run_model(image_path)
    if boxes:
        bbox_image_path = draw_boxes(image_path, boxes)
        return jsonify({
            'success': True,
            'bbox_image': f'/image/{os.path.basename(bbox_image_path)}',
            'boxes': boxes
        })
    
    return jsonify({'success': False})

@app.route('/reject/<image_id>', methods=['POST'])
def reject_image(image_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM images WHERE id = ?", (image_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/save_bboxes/<image_id>', methods=['POST'])
def save_bounding_boxes(image_id):
    boxes = request.json.get('boxes', [])
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT original_path FROM images WHERE id = ?", (image_id,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return jsonify({'success': False})
    
    filename = result[0]
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    cropped_paths = crop_and_save_bboxes(image_path, boxes)
    
    for cropped_path, x, y, w, h in cropped_paths:
        save_bbox_to_db(image_id, cropped_path, x, y, w, h)
    
    return jsonify({'success': True})

@app.route('/get_gallery')
def get_gallery():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT original_path, created_at FROM images ORDER BY created_at DESC")
    original_images = c.fetchall()
    
    c.execute("SELECT bbox_path, created_at FROM bounding_boxes ORDER BY created_at DESC")
    bbox_images = c.fetchall()
    
    conn.close()
    
    original_data = []
    for path, created_at in original_images:
        if os.path.exists(path):
            original_data.append({
                'url': f'/image/{os.path.basename(path)}',
                'created_at': created_at
            })
    
    bbox_data = []
    for path, created_at in bbox_images:
        if os.path.exists(path):
            bbox_data.append({
                'url': f'/image/{os.path.basename(path)}',
                'created_at': created_at
            })
    
    return jsonify({
        'original_images': original_data,
        'bbox_images': bbox_data
    })

@app.route('/image/<filename>')
def get_image(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/jpeg')
    return '', 404

if __name__ == '__main__':
    init_db()
    load_onnx_model()
    load_YOLO_model()
    app.run(port=5001)