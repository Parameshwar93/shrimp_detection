import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import shutil
import pandas as pd

base_path = "/content/drive/MyDrive/Checktray1/train"
images_dir = os.path.join(base_path, "images")
labels_dir = os.path.join(base_path, "labels")

# Create target directories
train_images_dir = os.path.join(images_dir, "train")
val_images_dir = os.path.join(images_dir, "val")
train_labels_dir = os.path.join(labels_dir, "train")
val_labels_dir = os.path.join(labels_dir, "val")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get lists of files and sort them to keep them in order and aligned
all_images = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
all_labels = sorted([f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))])

# Split index calculations (75:25)
split_idx = int(len(all_images) * 0.75)

train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

print(f"Total images found: {len(all_images)}")
print(f"Splitting into {len(train_images)} train images and {len(val_images)} validation images.")

# Move images to respective directories
for img in train_images:
    shutil.move(os.path.join(images_dir, img), os.path.join(train_images_dir, img))
for img in val_images:
    shutil.move(os.path.join(images_dir, img), os.path.join(val_images_dir, img))

# Match labels and move them to their respective directories
for img in train_images:
    base_name = os.path.splitext(img)[0]
    label_file = f"{base_name}.txt"
    if os.path.exists(os.path.join(labels_dir, label_file)):
        shutil.move(os.path.join(labels_dir, label_file), os.path.join(train_labels_dir, label_file))

for img in val_images:
    base_name = os.path.splitext(img)[0]
    label_file = f"{base_name}.txt"
    if os.path.exists(os.path.join(labels_dir, label_file)):
        shutil.move(os.path.join(labels_dir, label_file), os.path.join(val_labels_dir, label_file))

print("Dataset splitting completed successfully!")


### Training Loop
# Load a pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Train the model on your custom dataset
# patience=5: activates early stopping if validation loss doesn't improve for 5 consecutive epochs
results = model.train(
    data="/content/drive/MyDrive/Checktray1/data.yaml",
    epochs=50,
    patience=5,
    val=True
)

# Post-training validation to ensure the Confusion Matrix is generated
metrics = model.val()
print("Validation completed!")

# Extract global metrics
print("\n=== Global Performance Metrics ===")
print(f"Mean Precision (all classes): {metrics.results_dict['metrics/precision(B)']:.4f}")
print(f"Mean Recall (all classes):    {metrics.results_dict['metrics/recall(B)']:.4f}")
print(f"mAP @ 0.50 (all classes):     {metrics.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP @ 0.50-0.95:              {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")

# Extract per-class metrics
print("\n=== Per-Class Performance Metrics ===")
class_names = model.names
class_data = []

for i, name in class_names.items():
    precision = metrics.box.p[i] if len(metrics.box.p) > i else 0.0
    recall = metrics.box.r[i] if len(metrics.box.r) > i else 0.0
    ap50 = metrics.box.ap50[i] if len(metrics.box.ap50) > i else 0.0
    ap = metrics.box.ap[i] if len(metrics.box.ap) > i else 0.0
    
    class_data.append({
        "Class Name": name,
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "AP @ 0.50": f"{ap50:.4f}",
        "AP @ 0.50-0.95": f"{ap:.4f}"
    })

# Displaying per-class metrics as a clean pandas DataFrame
df_metrics = pd.DataFrame(class_data)
import IPython.display as display
display.display(df_metrics)

                          ## Inference with gradio ##

# Path to the best model weights
model_path = "/content/drive/MyDrive/Checktray1/best.pt"
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
