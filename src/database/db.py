import sqlite3
from src.config import *

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