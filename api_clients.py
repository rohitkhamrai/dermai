import requests
import json
import os

# --- Configuration ---
# Your personal OpenRouter API key is included as requested.
OPENROUTER_API_KEY = "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" 

# --- API Details (Using the proven, working models) ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_NAMES = {
    "gatekeeper": "qwen/qwen2.5-vl-32b-instruct:free",
    "consultant": "qwen/qwen2.5-vl-32b-instruct:free",
    "scribe": "tngtech/deepseek-r1t2-chimera:free"
}

HTTP_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://lightning.ai",
    "X-Title": "AI Skin Lesion Analyzer"
}

def call_openrouter_api(model_name: str, messages: list):
    """A unified function to call any model on the OpenRouter API."""
    data = {"model": model_name, "messages": messages}
    try:
        response = requests.post(OPENROUTER_API_URL, headers=HTTP_HEADERS, data=json.dumps(data), timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"API request failed for model {model_name}: {e}\nResponse Body: {response.text}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Failed to parse response from model {model_name}: {e}")
        return None

def run_gatekeeper_check(image_base64: str):
    """Uses Qwen-VL via OpenRouter for triage and quality control."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "Analyze this image. Is the primary subject human skin? Is the quality good or poor? Respond ONLY with a valid JSON object with three keys: 'is_skin' (true/false), 'subject' (one-word description), and 'quality' ('good' or 'poor')."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}]
    response_content = call_openrouter_api(MODEL_NAMES["gatekeeper"], messages)
    if response_content:
        try:
            start_index = response_content.find('{')
            end_index = response_content.rfind('}') + 1
            if start_index != -1 and end_index != -1:
                json_string = response_content[start_index:end_index]
                return json.loads(json_string)
            raise json.JSONDecodeError("Could not find JSON object.", response_content, 0)
        except json.JSONDecodeError as e:
            print(f"Gatekeeper did not return valid JSON. Error: {e}. Raw response: {response_content}")
            return None
    return None

def get_expert_consultation(image_base64: str):
    """Uses Qwen-VL via OpenRouter for a detailed clinical description."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "You are a dermatological assistant AI. Describe the visual characteristics of the skin lesion in this image. Focus on color variegation, border irregularity, and surface texture. Capture micro characteristics, Provide a concise, clinical paragraph."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}]
    return call_openrouter_api(MODEL_NAMES["consultant"], messages) or "No visual description could be generated."

def generate_final_report(internal_prediction: dict, visual_description: str):
    """Uses Deepseek via OpenRouter to generate a final, human-readable report."""
    prompt = f"You are a medical scribe AI. Synthesize the following information into a clear, professional report. Internal AI Prediction: {internal_prediction['prediction']} with {internal_prediction['confidence']} confidence. Consultant AI Visual Description: \"{visual_description}\". Based ONLY on this data, write a one-paragraph summary. Start with the internal AI's prediction. Incorporate the visual description. Do NOT give medical advice. Conclude with the exact sentence: \"This is an AI-generated analysis and is not a substitute for a professional medical diagnosis. Please consult a qualified dermatologist for any health concerns.\""
    messages = [{"role": "user", "content": prompt}]
    return call_openrouter_api(MODEL_NAMES["scribe"], messages) or "The final report could not be generated."
