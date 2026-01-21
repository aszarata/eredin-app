from PIL import Image
import numpy as np

def run_product_model(image_path, fridges_yolo_model):
    # if not onnx_session:
    #     return None
    
    # img = Image.open(image_path).convert('RGB')
    # img_resized = img.resize((640, 640))
    # img_array = np.array(img_resized).astype(np.float32) / 255.0
    # img_array = img_array.transpose(2, 0, 1)  
    # img_array = img_array[np.newaxis, ...]

    # input_name = onnx_session.get_inputs()[0].name
    # outputs = onnx_session.run(None, {input_name: img_array})
    if not fridges_yolo_model:
        return None
    
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    outputs = fridges_yolo_model(img_array, conf=0.9)

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