from ultralytics import YOLO
import cv2

# Load pre-trained model
model = YOLO('yolov8n.pt')

# Path to your images
image_folder = r'C:\Users\Kevin\Downloads\aircraft_detector\aircraft_images'

# Lower confidence to detect more (default is 0.25)
# Set to 0.1 to be more sensitive
results = model.predict(
    source=image_folder,
    conf=0.1,      # Lower threshold = more detections
    save=True,     # Save annotated images
    save_txt=True,  # Save detection text files
    show=False     # Don't show pop-up windows
)

print("\n" + "="*50)
print("DETECTION RESULTS")
print("="*50)

total_detected = 0
total_images = len(results)

for i, j in enumerate(results):
    img_name = j.path.split('\\')[-1]
    boxes = j.boxes

    if len(boxes) > 0:
        total_detected += 1
        print(f"\n✅ {img_name}: DETECTED")

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"   - Class: {model.names[cls]}, Confidence: {conf:.2%}")
    else:
        print(f"\n❌ {img_name}: NOT DETECTED")

print("\n" + "="*50)
print(f"Summary: {total_detected}/{total_images} images detected")
print(f"Saved images to: runs/detect/predict/")
print("="*50)
