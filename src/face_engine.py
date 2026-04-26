import face_recognition
import cv2
import os
import glob
import numpy as np

class FaceEngine:
    def __init__(self, known_faces_dir="data/known_faces"):
        self.known_faces_dir = known_faces_dir
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """
        Scans the known_faces directory. Each top-level subfolder is treated 
        as a person's name. All images inside that folder (and its subfolders)
        are indexed as examples of that person.
        """
        print("👤 Loading known faces library...")
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            return

        # Get only the top-level directories
        for person_name in os.listdir(self.known_faces_dir):
            person_path = os.path.join(self.known_faces_dir, person_name)
            
            if not os.path.isdir(person_path):
                # Skip individual files in the root to encourage folder structure
                if person_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"ℹ️ Skipping root file '{person_name}'. Please put it in a folder.")
                continue

            # Process every image in this person's folder
            for root, _, files in os.walk(person_path):
                for file in files:
                    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        img = face_recognition.load_image_file(file_path)
                        encodings = face_recognition.face_encodings(img)
                        for enc in encodings:
                            self.known_encodings.append(enc)
                            self.known_names.append(person_name)
                        if len(encodings) > 0:
                            print(f"✅ Indexed: {person_name} (from {file})")
                    except Exception as e:
                        print(f"❌ Error loading {file_path}: {e}")

    def identify(self, frame):
        """
        Detects and identifies faces in a BGR frame (OpenCV format).
        Returns a list of tuples: (location, name)
        """
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        identities = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, encoding, tolerance=0.6)
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_encodings, encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
            
            identities.append(name)
            
        return list(zip(face_locations, identities))

if __name__ == "__main__":
    # Test script
    engine = FaceEngine()
    test_img_path = "/Users/princegondaliya/.gemini/tmp/cctv/images/clipboard-1777231819193.png"
    
    if os.path.exists(test_img_path):
        print(f"📸 Testing face detection on: {test_img_path}")
        frame = cv2.imread(test_img_path)
        if frame is not None:
            results = engine.identify(frame)
            print("\n--- FACE RESULTS ---")
            for loc, name in results:
                print(f"👤 Identity: {name} | Location: {loc}")
            if not results:
                print("❌ No faces detected in the image.")
        else:
            print("❌ Failed to read image file.")
    else:
        print(f"❌ Image not found at {test_img_path}")
