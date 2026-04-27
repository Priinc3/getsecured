import ollama
import json
import base64
import cv2
import os
import re
from groq import Groq

class VLMEngine:
    def __init__(self, model="llama-3.2-11b-vision-preview", provider="local", api_key=None):
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.version = "4.0.0-groq-integrated"
        self.client = Groq(api_key=api_key) if provider == "groq" and api_key else None
        print(f"🚀 VLM Engine v{self.version} Initialized ({self.provider})")

    def describe_event(self, frame, identity="Unknown", detections=[], last_report=None):
        """
        Sends the frame to a VLM (Local or Groq) and parses the response.
        """
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        objects_str = ", ".join(detections) if detections else "general environment"
        memory_context = f"\n[PREVIOUS STATE]: {last_report}" if last_report else ""
        
        prompt = f"""
        [SYSTEM REFERENCE DATA]{memory_context}
        Verified Subject: {identity}
        Verified Detections: {objects_str}
        
        [INSTRUCTIONS]
        Analyze the image and the System Data. Use the data to describe exactly what is happening.
        If there is a [PREVIOUS STATE], update the story based on new movements.
        
        Provide a security report with these exact headers:
        ACTION: Detail what is happening now.
        CONTEXT: Describe the setting.
        ITEMS: List any dangerous or suspicious items.
        ALERT: (Low, Medium, High, or Critical)
        SUMMARY: A one-sentence security overview.
        """

        if self.provider == "groq":
            return self._call_groq(img_base64, prompt)
        else:
            return self._call_ollama(img_base64, prompt)

    def _call_ollama(self, img_base64, prompt):
        print(f"🧠 [Local] Sending to {self.model}...")
        try:
            response = ollama.generate(model=self.model, prompt=prompt, images=[img_base64], options={"temperature": 0.1})
            return self._parse_response(response.get('response', '').strip())
        except Exception as e:
            return {"action": "Error", "summary": f"Local VLM Error: {str(e)}"}

    def _call_groq(self, img_base64, prompt):
        print(f"⚡ [Groq-Scout] Sending to {self.model}...")
        if not self.client:
            return {"action": "Error", "summary": "Groq API Key missing."}
        
        try:
            # Using the specific Llama 4 Scout model provided by the user
            completion = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False # Changed to False for internal parsing
            )
            raw_text = completion.choices[0].message.content
            return self._parse_response(raw_text)
        except Exception as e:
            print(f"❌ Groq Error: {e}")
            return {"action": "Error", "summary": f"Groq API Error: {str(e)}"}

    def _parse_response(self, raw_text):
        print(f"📝 RAW VLM RESPONSE:\n{raw_text}")
        
        parsed = {
            "action": self._extract(raw_text, r"(?:ACTION|Action|action):\s*(.*)"),
            "context": self._extract(raw_text, r"(?:CONTEXT|Context|context):\s*(.*)"),
            "suspicious_objects": [],
            "summary": self._extract(raw_text, r"(?:SUMMARY|Summary|summary):\s*(.*)"),
            "alert_level": "Low",
            "alert_type": "None"
        }

        # --- GLOBAL THREAT HEURISTIC ---
        full_text_lower = raw_text.lower()
        weapons = ["gun", "pistol", "rifle", "firearm", "knife", "blade", "sword", "machete", "axe"]
        found_weapons = [w for w in weapons if w in full_text_lower]
        
        suspicious = ["mask", "balaclava", "crowbar", "hammer", "breaking", "forced", "climbing"]
        found_suspicious = [s for s in suspicious if s in full_text_lower]

        if found_weapons:
            parsed["alert_level"] = "Critical"
            parsed["alert_type"] = f"CRITICAL: Weapon ({', '.join(found_weapons)})"
        elif found_suspicious:
            parsed["alert_level"] = "High"
            parsed["alert_type"] = f"HIGH: Suspicious ({', '.join(found_suspicious)})"

        if not parsed["summary"] or parsed["summary"] == "Unknown":
            parsed["summary"] = raw_text.split('\n')[0]
            parsed["action"] = "Activity Detected"
        
        return parsed

    def _extract(self, text, pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        if match: return match.group(1).split('\n')[0].strip()
        return "Unknown"

if __name__ == "__main__":
    engine = VLMEngine()
    print("💡 VLM Engine v4.0 (Groq Ready) Initialized.")
