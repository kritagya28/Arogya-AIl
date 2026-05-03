import requests
import json

# This is the URL where your server is listening
url = 'http://127.0.0.1:5000/diagnose'

# This is our fake patient's data
patient_data = {
    "symptoms": "severe joint pain rash slight fever",
    "dosha": "Pitta",
    "season": "Summer"
}

print("Sending patient data to Arogya-AI...")

# Send the data to your app.py server
response = requests.post(url, json=patient_data)

# Print the result!
result = response.json()
print("\n--- AROGYA-AI RESULTS ---")
print(f"ML DETECTOR PREDICTION: {result['ml_raw_prediction']}")
print(f"LLM EXPLANATION:\n{result['ayurvedic_report']}")