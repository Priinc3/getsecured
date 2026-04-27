import json
from groq import Groq
import os

class WatchdogAgent:
    """
    The Watchdog Agent is the decision-making core of the Smart CCTV Investigator.
    It analyzes detections, identities, and AI narratives to assess security risks.
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def assess_threat(self, detections, identities, vlm_summary):
        """
        Analyzes the scene data and returns a structured threat assessment.
        """
        if not self.client:
            return self._fallback_assessment(detections, identities)

        prompt = f"""
        System: You are the 'Watchdog' security agent for a high-end AI CCTV system.
        Your task is to analyze detection data and provide a threat score (0-100) and an alert status.
        
        Input Data:
        - Detections (YOLO): {detections}
        - Recognized Identities: {identities}
        - AI Scene Description: {vlm_summary}
        
        Rules:
        1. If a weapon (gun, knife) is detected, Threat Score is at least 80.
        2. If an 'Unknown' person is detected in a restricted area, Threat Score is at least 50.
        3. If a known person (e.g., Akshay, Prince) is present without weapons, Threat Score is < 20.
        
        Return ONLY a JSON object with the following keys:
        - score (int: 0-100)
        - level (string: Low, Medium, High, Critical)
        - reason (string: Brief explanation)
        - action (string: Recommended immediate action)
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Watchdog Error: {e}")
            return self._fallback_assessment(detections, identities)

    def _fallback_assessment(self, detections, identities):
        """
        Basic rule-based assessment if the LLM is unavailable.
        """
        score = 0
        reason = "Routine monitoring."
        action = "None."
        
        has_weapon = any(d in ["gun", "knife"] for d in detections)
        has_unknown = "Unknown" in identities or not identities
        
        if has_weapon:
            score = 90
            level = "Critical"
            reason = "Weapon detected in scene."
            action = "Trigger alarm and notify authorities."
        elif has_unknown and "person" in detections:
            score = 50
            level = "Medium"
            reason = "Unidentified person detected."
            action = "Monitor closely."
        else:
            level = "Low"
            
        return {
            "score": score,
            "level": level,
            "reason": reason,
            "action": action
        }
