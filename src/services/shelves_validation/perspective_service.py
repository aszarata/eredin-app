import cv2
import numpy as np

def find_horizontal_lines(img, min_length_ratio=0.4, angle_eps=50):
    height, width = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 120)

    min_line_length = int(width * min_length_ratio)
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=200,
        minLineLength=min_line_length,
        maxLineGap=width*0.1
    )

    return filter_horizontal_lines(lines, min_line_length, angle_eps)
    
def filter_horizontal_lines(lines, min_line_length, angle_eps):
    filtered_angles = []
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            if dx > 0:
                angle = np.abs(np.arctan(dy/dx) * 180/np.pi)
            else:
                angle = 90
            
            if angle < angle_eps:
                length = np.sqrt(dx**2 + dy**2)
                if length >= min_line_length:
                    filtered_angles.append(angle)
                    filtered_lines.append(line)
    
    return filtered_lines, filtered_angles