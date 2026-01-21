import cv2
import numpy as np

from src.services.shelves_validation.perspective_service import find_horizontal_lines

def run_shelves_pipeline(image_path, shelves_yolo_model):
    if not shelves_yolo_model:
        raise ValueError("No shelves detection model found.")
    
    img = cv2.imread(image_path)
    img_array = np.array(img)
    outputs = shelves_yolo_model(img_array, conf=0.15)

    cropped_img, closest_box = crop_image(outputs, img)

    lines, angles = find_horizontal_lines(cropped_img)
    return cropped_img, closest_box, lines, angles


def crop_image(outputs, img):
    if len(outputs[0].boxes) > 0:
        boxes = outputs[0].boxes.xyxy.cpu().numpy()
        img_center = np.array([img.shape[1]/2, img.shape[0]/2])
        
        box_centers = np.array([[(box[0]+box[2])/2, (box[1]+box[3])/2] for box in boxes])
        distances = np.linalg.norm(box_centers - img_center, axis=1)
        closest_idx = np.argmin(distances)
        
        closest_box = boxes[closest_idx].astype(int)
        x1, y1, x2, y2 = closest_box

        cropped_img = img[y1:y2, x1:x2]

        return cropped_img, closest_box
    else:
        raise ValueError("No shelves have been detected on the photo.")