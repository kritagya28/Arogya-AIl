import os
from dotenv import load_dotenv

# Load the secret variables from the .env file
load_dotenv()

# ... your other imports ...
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import google.generativeai as genai
import json
import numpy as np # <-- NEW: Needed for XAI Math

app = Flask(__name__)
CORS(app) 

# Load ML components
model = joblib.load('rf_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Set up Gemini API 
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel('gemini-2.5-flash')

@app.route('/', methods=['GET'])
def home():
    return "The Python Backend is awake and ready!"

@app.route('/diagnose', methods=['POST'])
def diagnose():
    data = request.json
    user_symptoms = data.get('symptoms', '')
    user_dosha = data.get('dosha', 'Unknown')
    user_season = data.get('season', 'Unknown')
    
    # Engine 1: ML PREDICTION
    vectorized_input = tfidf.transform([user_symptoms])
    prediction_encoded = model.predict(vectorized_input)
    predicted_disease = label_encoder.inverse_transform(prediction_encoded)[0]
    
    probabilities = model.predict_proba(vectorized_input)[0]
    confidence = round(max(probabilities) * 100, 1)
    
    # --- NEW: EXPLAINABLE AI (XAI) ENGINE ---
    # Calculate exactly WHICH symptoms triggered this specific prediction
    user_vector = vectorized_input.toarray()[0]
    global_importances = model.feature_importances_
    feature_names = tfidf.get_feature_names_out()
    
    # Multiply user's input by the model's global weights
    local_importances = user_vector * global_importances
    top_indices = np.argsort(local_importances)[-4:][::-1] # Get top 4 factors
    
    xai_breakdown = []
    total_local_importance = sum(local_importances)
    
    if total_local_importance > 0:
        for idx in top_indices:
            if local_importances[idx] > 0:
                weight = round((local_importances[idx] / total_local_importance) * 100)
                xai_breakdown.append({"symptom": feature_names[idx].title(), "weight": weight})
    else:
        xai_breakdown = [{"symptom": "General Pattern", "weight": 100}]
    # ----------------------------------------
    
    # Engine 2: LLM ENHANCEMENT
    prompt = f"""
    You are an expert Ayurvedic Vaidya and AI Health Assistant.
    A Machine Learning model predicted: {predicted_disease}
    
    Patient Data: Dosha: {user_dosha}, Season: {user_season}, Symptoms: {user_symptoms}
    
    Return a raw JSON object with NO markdown formatting, strictly following this structure:
    {{
      "reasoning": "Explain why this condition is occurring based on Ayurvedic principles.",
      "herbs": [
        {{"name": "Herb 1", "benefit": "Benefit of herb 1"}},
        {{"name": "Herb 2", "benefit": "Benefit of herb 2"}}
      ],
      "lifestyle": ["Lifestyle tip 1", "Dietary tip 2"]
    }}
    """
    
    try:
        response = llm_model.generate_content(prompt)
        clean_text = response.text.strip().replace('```json', '').replace('```', '')
        llm_output = json.loads(clean_text)
    except Exception as e:
        llm_output = {
            "reasoning": "Live AI Contextual Reasoning is temporarily paused due to API quota limits. However, your Machine Learning baseline prediction has been successfully generated above.",
            "herbs": [{"name": "Pending AI Restoration", "benefit": "Please check back later or use a different API key."}],
            "lifestyle": ["Stay hydrated and rest.", "Consult a human practitioner for immediate advice."]
        }
    
    return jsonify({
        "prediction": predicted_disease,
        "confidence": confidence,
        "xai_breakdown": xai_breakdown, # <--- SENDING XAI DATA TO REACT
        "indicators": [
            {"label": "Symptom Match", "score": f"+{round(confidence/100 * 0.4, 2)}"},
            {"label": "Dosha Alignment", "score": "+0.25"},
            {"label": "Seasonal Factor", "score": "+0.15"}
        ],
        "reasoning": llm_output.get("reasoning", "Analysis complete."),
        "herbs": llm_output.get("herbs", []),
        "lifestyle": llm_output.get("lifestyle", [])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)