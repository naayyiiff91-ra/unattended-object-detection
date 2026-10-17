import os
import base64
import requests
from dotenv import load_dotenv

# Load GEMINI_API_KEY from .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Gemma 4 multimodal model identifier
MODEL_ID = "gemma-4-26b-a4b-it"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"


def _clean_alert_output(raw_text):
    """Extracts the final alert line from Gemma 4's response."""
    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    # Extract line prefixed with ALERT: if present
    for line in reversed(lines):
        if "ALERT:" in line:
            return line.split("ALERT:", 1)[1].strip().strip('"\'* ')
    # Fallback to the last non-empty line
    return lines[-1].strip().strip('"\'* ')


def generate_alert(bag_id, class_name, duration_seconds, zone="the monitored area", snapshot_path=None):
    """
    Sends event details (and optional snapshot image) to Gemma 4 via the Gemini API
    and returns a natural-language security alert.
    Falls back to a template if the API fails or GEMINI_API_KEY is not set.
    """
    duration_minutes = duration_seconds / 60
    duration_str = f"{int(duration_seconds)} seconds" if duration_seconds < 60 else f"{duration_minutes:.1f} minutes"

    # Reliable fallback message
    fallback_message = (
        f"Security Alert: An unattended {class_name} (ID {bag_id}) has been left in {zone} "
        f"for approximately {duration_str}."
    )

    if not API_KEY:
        print("[GEMMA] GEMINI_API_KEY not configured in .env, using fallback template.")
        return fallback_message

    parts = []

    # If a snapshot image is provided and exists, attach it as inline data for Gemma 4 vision
    if snapshot_path and os.path.exists(snapshot_path):
        try:
            with open(snapshot_path, "rb") as img_file:
                b64_image = base64.b64encode(img_file.read()).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64_image
                }
            })
            prompt = (
                f"You are a surveillance AI. An unattended {class_name} (tracking ID {bag_id}) was detected in {zone} "
                f"left without an owner for {duration_str}. "
                f"Analyze the image and write a single, professional security alert sentence describing the object's appearance and location. "
                f"On the very last line, output only: ALERT: <your single sentence>."
            )
        except Exception as e:
            print(f"[GEMMA IMAGE ERROR] Failed to attach snapshot: {e}")
            prompt = (
                f"Write a single, concise security alert sentence for a surveillance system. "
                f"An unattended {class_name} (tracking ID {bag_id}) has been left in {zone} "
                f"for approximately {duration_str}. "
                f"On the very last line, output only: ALERT: <your single sentence>."
            )
    else:
        # Text-only prompt
        prompt = (
            f"Write a single, concise security alert sentence for a surveillance system. "
            f"An unattended {class_name} (tracking ID {bag_id}) has been left in {zone} "
            f"for approximately {duration_str}. "
            f"On the very last line, output only: ALERT: <your single sentence>."
        )

    parts.append({"text": prompt})

    try:
        url = f"{GEMINI_URL}?key={API_KEY}"
        payload = {"contents": [{"parts": parts}]}
        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()

        data = response.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        alert_text = _clean_alert_output(raw_text)

        if alert_text:
            return alert_text
        else:
            raise ValueError("Empty response from Gemma 4")

    except Exception as e:
        print(f"[GEMMA API ERROR] Falling back to template alert: {e}")
        return fallback_message