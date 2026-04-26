import ollama
import base64
import os

def test_vlm():
    model = "moondream"
    img_path = "data/videos/1000085346_frame_00476_jpg.rf.eefdb567d0654fbf5d14927f699883bf.jpg"
    
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return

    with open(img_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')

    print(f"📡 Testing {model} with raw prompt...")
    try:
        # Step 1: Just a plain description (No JSON)
        res = ollama.generate(
            model=model,
            prompt="What is the person in this image doing? Describe the scene.",
            images=[img_base64]
        )
        print("\n--- PLAIN RESPONSE ---")
        print(res['response'])
        
        # Step 2: Try JSON
        print("\n📡 Testing JSON formatting...")
        res_json = ollama.generate(
            model=model,
            prompt="Describe this image. Return as JSON with key 'description'.",
            images=[img_base64],
            format="json"
        )
        print("\n--- JSON RESPONSE ---")
        print(res_json['response'])

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_vlm()
