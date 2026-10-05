import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Path to the best model weights
model_path = "/content/drive/MyDrive/Checktray1/best_model_weights.pt"

# Load the YOLO model with the weights
model = YOLO(model_path)

print("Model weights loaded successfully!")

# Load the custom trained YOLO model
model_path = "/content/drive/MyDrive/Checktray1/shrimp_weights.pt"
model = YOLO(model_path)

def detect_shrimp(input_img):
    if input_img is None:
        return None, 0
    
    # Run inference on the input image
    # conf: confidence threshold (e.g., 0.25 means detect anything >= 25% confidence)
    # iou: NMS Intersection over Union threshold (e.g., 0.60 loosens overlap boundaries)
    results = model(input_img, conf=0.25, iou=0.60)
    
    # Obtain class names
    class_names = model.names
    
    # Find the index/class ID of the 'shrimp' class (case-insensitive check)
    shrimp_class_ids = [idx for idx, name in class_names.items() if "shrimp" in name.lower()]
    
    shrimp_count = 0
    
    # If the class exists in the model training classes, filter detections
    if shrimp_class_ids:
        # Iterate through the results and filter classes
        for r in results:
            # Keep only the boxes that match the 'shrimp' class ID(s)
            mask = np.isin(r.boxes.cls.cpu().numpy(), shrimp_class_ids)
            r.boxes = r.boxes[mask]
            shrimp_count += len(r.boxes)
    
    # Plot the filtered results on the image
    annotated_img = results[0].plot()
    
    # Convert BGR (OpenCV format) to RGB
    annotated_img_rgb = Image.fromarray(annotated_img[..., ::-1])
    
    return annotated_img_rgb, shrimp_count

# Define Gradio interface with an additional output for shrimp count
demo = gr.Interface(
    fn=detect_shrimp,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Shrimp Detections"),
        gr.Number(label="Detected Shrimp Count", precision=0)
    ],
    title="Shrimp Object Detection & Counting App",
    description="Upload an image to detect and count 'shrimp' objects using your custom-trained YOLO model."
)

# Launch the Gradio app and share it publicly
demo.launch(share=True)
