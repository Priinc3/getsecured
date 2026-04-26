from ultralytics import YOLOWorld
import cv2
import os

def initialize_vision_model():
    """
    Initializes the YOLO-World model with custom classes for the CCTV investigator.
    """
    print("🚀 Initializing YOLO-World model...")
    # Load a pre-trained YOLO-World model (Medium is a good balance for MVP)
    model = YOLOWorld('yolov8m-worldv2.pt') 

    # Define the specific classes we want to monitor
    # These are 'prompted' into the model
    custom_classes = [
        "face", "car", "bicycle", "motorcycle", "person", 
        "cell phone", "laptop", "knife", "gun"
    ]
    model.set_classes(custom_classes)
    
    # Save the model with the custom classes baked in for faster loading next time
    model.save("models/custom_yolo_world.pt")
    print(f"✅ Model initialized and saved with classes: {custom_classes}")
    return model

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # Initialize/Load model
    model = initialize_vision_model()
    
    # Path to the test image
    image_path = "/Users/princegondaliya/.gemini/tmp/cctv/images/clipboard-1777231819193.png"
    
    if os.path.exists(image_path):
        print(f"📸 Running inference on: {image_path}")
        # Lowering conf to 0.15 to catch potential threats in low-light CCTV
        results = model.predict(image_path, save=True, project="data/output", name="cctv_test", conf=0.15)
        
        # Parse and print results
        print("\n--- DETECTION RESULTS ---")
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                print(f"🔍 Found: {label} (Confidence: {conf:.2f})")
        print("--------------------------")
        print(f"✅ Processed image saved to data/output/test_inference/")
    else:
        print(f"❌ Could not find image at {image_path}")
