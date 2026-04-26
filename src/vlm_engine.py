import ollama
import json
import base64
import cv2
import os
import re

class VLMEngine:
    def __init__(self, model="moondream"):
        self.model = model
        self.version = "2.1.0-smart-parsing"
        print(f"🚀 VLM Engine v{self.version} Initialized")

    def describe_event(self, frame, identity="Unknown", detections=[]):
        """
        Sends the frame to a local VLM and parses the response robustly.
        """
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 2. Prepare the Authoritative Prompt
        objects_str = ", ".join(detections) if detections else "general environment"
        prompt = f"""
        [SYSTEM REFERENCE DATA - DO NOT IGNORE]
        Verified Subject: {identity}
        Verified Detections: {objects_str}
        
        [INSTRUCTIONS]
        Use the System Reference Data above to analyze this image. 
        Describe the interaction between the Subject and the Detections.
        
        Provide a security report with these exact headers:
        ACTION: Detail what {identity} is doing with the {objects_str}.
        CONTEXT: Describe the setting and background.
        ITEMS: List any dangerous or suspicious items confirmed in the image.
        ALERT: (Low, Medium, High, or Critical)
        SUMMARY: A one-sentence security overview.
        """

        print(f"🧠 Sending event to VLM ({self.model})...")
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                images=[img_base64],
                options={"temperature": 0.1}
            )
            
            raw_text = response.get('response', '').strip()
            print(f"📝 RAW VLM RESPONSE:\n{raw_text}")
            
            if not raw_text:
                return {"action": "No response", "context": "N/A", "suspicious_objects": [], "summary": "VLM returned an empty response."}

            # --- SMART PARSING (Regex) ---
            parsed = {
                "action": self._extract(raw_text, r"(?:ACTION|Action|action):\s*(.*)"),
                "context": self._extract(raw_text, r"(?:CONTEXT|Context|context):\s*(.*)"),
                "suspicious_objects": [],
                "summary": self._extract(raw_text, r"(?:SUMMARY|Summary|summary):\s*(.*)"),
                "alert_level": "Low", # Default
                "alert_type": "None"
            }

            # --- GLOBAL THREAT HEURISTIC (Scan whole response) ---
            full_text_lower = raw_text.lower()
            
            # 1. Critical Threats: Lethal Weapons
            weapons = [
                "gun", "pistol", "rifle", "firearm", "shotgun", "handgun", "revolver",
                "knife", "dagger", "blade", "sword", "machete", "axe", "cleaver"
            ]
            found_weapons = [w for w in weapons if w in full_text_lower]
            
            # 2. High Threats: Break-in tools, disguises, and forced entry
            suspicious = [
                "mask", "balaclava", "ski mask", "hoodie", "face cover", 
                "crowbar", "hammer", "bolt cutter", "screwdriver", "picklock",
                "breaking", "forced", "smashed", "climbing", "crawling", "sneaking",
                "jumped", "fence", "window"
            ]
            found_suspicious = [s for s in suspicious if s in full_text_lower]

            if found_weapons:
                parsed["alert_level"] = "Critical"
                parsed["alert_type"] = f"CRITICAL: Lethal Weapon ({', '.join(found_weapons)})"
                if not parsed["suspicious_objects"]: parsed["suspicious_objects"] = found_weapons
            elif found_suspicious:
                parsed["alert_level"] = "High"
                parsed["alert_type"] = f"HIGH: Suspicious Activity ({', '.join(found_suspicious)})"
                if not parsed["suspicious_objects"]: parsed["suspicious_objects"] = found_suspicious

            # 3. Final Override: If AI explicitly says Critical/High, respect it
            ai_alert = self._extract(raw_text, r"(?:ALERT|Alert|alert):\s*(.*)").capitalize()
            if ai_alert in ["High", "Critical"] and parsed["alert_level"] == "Low":
                parsed["alert_level"] = ai_alert
                parsed["alert_type"] = "AI Flagged Threat"

            # --- FALLBACK LOGIC ---
            if not parsed["action"] or parsed["action"] == "Unknown":
                parsed["summary"] = raw_text.split('\n')[0]
                parsed["action"] = "Activity Detected"
            
            # Clean up
            for key in ["action", "context", "summary"]:
                if not parsed[key] or parsed[key].strip() == "" or parsed[key] == "Unknown":
                    parsed[key] = "Detailed in summary" if key != "summary" else raw_text

            return parsed
        except Exception as e:
            print(f"❌ VLM Error: {e}")
            return {"action": "Error", "context": "N/A", "suspicious_objects": [], "summary": str(e)}

    def _extract(self, text, pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            line = match.group(1).split('\n')[0].strip()
            return line
        return "Unknown"

if __name__ == "__main__":
    engine = VLMEngine()
    print("💡 VLM Engine Initialized. Ready to process frames.")
