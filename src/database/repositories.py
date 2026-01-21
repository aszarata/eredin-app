import uuid
from datetime import datetime
import sqlite3
from src.config import *

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