import cv2
from src.config import SHELVES_MAX_ANGLE, SHELVES_MAX_MARGIN

def run_shelves_checks(img_path, closest_box, angles):
    img = cv2.imread(img_path)
    check_too_close(img, closest_box)
    check_perspective(perspective_angles=angles)
    

def check_too_close(img, closest_box, margin_threshold=SHELVES_MAX_MARGIN):
    img_height, img_width = img.shape[:2]
    x1, y1, x2, y2 = closest_box
    if (y1 < margin_threshold or 
        img_height - y2 < margin_threshold):
        raise ValueError("You are too close to the shelves.")
    
def check_perspective(perspective_angles, max_angle=SHELVES_MAX_ANGLE):
    for angle in perspective_angles:
        if angle > max_angle or angle < -max_angle:
            raise ValueError("Wrong perspective! Stand in the front of the shelves.")