import uuid
import cv2
from src.config import *

def draw_boxes_or_lines_to_file(image_path, boxes, lines=None):
    img = cv2.imread(image_path)
    processed_img = draw_boxes(img, boxes)
    if lines and len(boxes) > 0:
        box_x1, box_y1, box_x2, box_y2 = boxes[0]
        processed_img = draw_lines(processed_img, lines, box_x1, box_y1)

    output_path = f"{UPLOAD_FOLDER}/bbox_{uuid.uuid4()}.jpg"
    cv2.imwrite(output_path, processed_img)
    return output_path

def draw_boxes(img, boxes):
    box_img = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(box_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return box_img

def draw_lines(img, lines, offset_x=0, offset_y=0):
    lines_img = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lines_img, 
                 (x1 + offset_x, y1 + offset_y), 
                 (x2 + offset_x, y2 + offset_y), 
                 (0, 0, 255), 3)
    return lines_img

def crop_and_save_bboxes(image_path, boxes):
    img = cv2.imread(image_path)
    cropped_paths = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cropped = img[y1:y2, x1:x2]
        if cropped.size > 0:
            cropped_path = f"{UPLOAD_FOLDER}/cropped_{uuid.uuid4()}.jpg"
            cv2.imwrite(cropped_path, cropped)
            cropped_paths.append((cropped_path, x1, y1, x2-x1, y2-y1))
    
    return cropped_paths