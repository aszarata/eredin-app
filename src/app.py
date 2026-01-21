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

from src.config import *
from src.database.db import init_db
from src.database.repositories import save_bbox_to_db, save_image_to_db
from src.services.model_loader import load_onnx_model, load_fridges_YOLO_model, load_shelves_YOLO_model
from src.services.image_services import draw_boxes_or_lines_to_file, crop_and_save_bboxes
from src.services.shelves_validation.shelves_detection_service import run_shelves_pipeline
from src.services.shelves_validation.shelves_checks import run_shelves_checks
from src.services.product_detection.product_detection_service import run_product_model

app = Flask(__name__)

onnx_session = None
fridges_yolo_model = None
shelves_yolo_model = None

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

    img_url = f'/image/{filename}'
    image_problem = None
    try:
        cropped_image, closest_box, lines, angles = run_shelves_pipeline(filepath, shelves_yolo_model)
        bbox_image_path = draw_boxes_or_lines_to_file(filepath, [closest_box], lines)
        img_url = f'/image/{os.path.basename(bbox_image_path)}'

        run_shelves_checks(img_path=filepath, closest_box=closest_box, angles=angles)
        cv2.imwrite(filepath, cropped_image)

    except ValueError as e:
        image_problem = str(e)

    return jsonify({
        'success': True,
        'image_id': image_id,
        'image_url': img_url,
        'image_problem': image_problem
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
    boxes = run_product_model(image_path, fridges_yolo_model)
    if boxes:
        bbox_image_path = draw_boxes_or_lines_to_file(image_path, boxes)
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
    onnx_session = load_onnx_model()
    fridges_yolo_model = load_fridges_YOLO_model()
    shelves_yolo_model = load_shelves_YOLO_model()
    app.run(port=5001)