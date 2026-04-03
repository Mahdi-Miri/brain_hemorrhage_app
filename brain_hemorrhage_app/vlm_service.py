import requests
import base64

# =========================
# GROQ INFERENCE API (Blazing Fast & Free)
# =========================
API_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_API_KEY = "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"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def analyze_with_vlm(image_bytes: bytes, api_result: str, probability: str):
    try:
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        prompt = f"""
You are a senior neuroradiologist.
A deep learning model analyzed this CT scan:
- Prediction: {api_result}
- Probability: {probability}

Now analyze the image carefully:
1. Is intracranial hemorrhage present?
2. If yes, what type?
3. Do you agree with the DL model?
4. Short reasoning.
"""

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 800,
            "temperature": 0.2 
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']

            return f"""
### 🧠 VLM Report (Powered by Groq)

**DL Result:** {api_result}  
**Confidence:** {probability}

---

**🧾 Analysis:**
{answer}
"""
        else:
            return f"❌ Groq API Error {response.status_code}: {response.text}"

    except Exception as e:
        return f"❌ Exception during VLM analysis: {e}"