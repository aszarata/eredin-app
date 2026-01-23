import cv2
import numpy as np

def find_horizontal_lines(img, min_length_ratio=0.4, angle_eps=50, target_size=640):
    orig_height, orig_width = img.shape[:2]
    
    scale_factor, scaled_img = scale_image_to_target(img, target_size)
    scaled_height, scaled_width = scaled_img.shape[:2]
    
    gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 120)
    
    min_line_length_scaled = int(scaled_width * min_length_ratio)
    
    lines_scaled = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=200,
        minLineLength=min_line_length_scaled,
        maxLineGap=scaled_width*0.1
    )
    
    filtered_lines_scaled, filtered_angles = filter_horizontal_lines(
        lines_scaled, min_line_length_scaled, angle_eps
    )
    
    filtered_lines_original = scale_lines_to_original(
        filtered_lines_scaled, scale_factor
    )
    
    return filtered_lines_original, filtered_angles

def scale_image_to_target(img, target_size=640):
    height, width = img.shape[:2]
    
    if height > width:
        scale_factor = target_size / height
        new_height = target_size
        new_width = int(width * scale_factor)
    else:
        scale_factor = target_size / width
        new_width = target_size
        new_height = int(height * scale_factor)
    
    scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return scale_factor, scaled_img

def scale_lines_to_original(lines_scaled, scale_factor):
    if lines_scaled is None or len(lines_scaled) == 0:
        return []
    
    lines_original = []
    inv_scale_factor = 1.0 / scale_factor
    
    for line in lines_scaled:
        x1, y1, x2, y2 = line[0]
        x1_orig = int(x1 * inv_scale_factor)
        y1_orig = int(y1 * inv_scale_factor)
        x2_orig = int(x2 * inv_scale_factor)
        y2_orig = int(y2 * inv_scale_factor)
        
        lines_original.append([[x1_orig, y1_orig, x2_orig, y2_orig]])
    
    return lines_original

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