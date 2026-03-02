from ultralytics import YOLO
import cv2

# Load the pre-trained model
model = YOLO('yolov8n.pt')

# Open the webcam
# 0 = default camera
cap = cv2.VideoCapture(0)

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Starting webcam detection...")
print("Press 'q' to quit")

# Set window name
window_name = 'Airplane Detection'

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

 # Run detection on this frame
    # conf=0.2 lowers threshold to detect more
    results = model.predict(
        source=frame,
        conf=0.2,
        verbose=False  # Disable verbose output
    )

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow(window_name, annotated_frame)
  # Print detection info
    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"Detected: {model.names[cls]} ({conf:.1%})")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
