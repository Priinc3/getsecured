import ollama
import json
import base64
import cv2
import os

class VLMEngine:
    def __init__(self, model="moondream"):
        self.model = model

    def describe_event(self, frame, identity="Unknown", detections=[]):
        """
        Sends the frame to a local VLM (via Ollama) to get a natural language description.
        """
        # 1. Convert frame to Base64 for Ollama
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 2. Prepare the prompt
        objects_str = ", ".join(detections) if detections else "various objects"
        prompt = f"""
        Analyze this image. A person identified as '{identity}' was detected.
        Other objects visible: {objects_str}.
        
        Describe the following in a concise summary:
        1. What is the person doing? (Action)
        2. Where are they? (Location Context)
        3. Are they holding anything suspicious?
        
        Return ONLY a JSON object with these keys: 
        {{
            "action": "string",
            "context": "string",
            "suspicious_objects": ["list"],
            "summary": "one sentence description"
        }}
        """

        print(f"🧠 Sending event to VLM ({self.model})...")
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                images=[img_base64],
                format="json", # Force JSON output
                options={"temperature": 0.2} # Low temp for factual reporting
            )
            
            # Parse response
            result = json.loads(response['response'])
            return result
        except Exception as e:
            print(f"❌ VLM Error: {e}")
            return {
                "action": "Error during analysis",
                "context": "N/A",
                "suspicious_objects": [],
                "summary": f"Could not analyze: {str(e)}"
            }

if __name__ == "__main__":
    # Test block
    # Note: Requires ollama running and 'moondream' pulled
    engine = VLMEngine()
    print("💡 VLM Engine Initialized. Ready to process frames.")
