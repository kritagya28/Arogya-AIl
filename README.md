# 🌿 ArogyaAI: Hybrid Intelligence for Ayurvedic Clinical Support

![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Firebase](https://img.shields.io/badge/firebase-%23039BE5.svg?style=for-the-badge&logo=firebase)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)

ArogyaAI is a modern, cloud-connected Clinical Decision Support System (CDSS) designed to bridge the gap between traditional Ayurvedic medicine and modern Artificial Intelligence. 

By utilizing a **Dual-Engine AI Architecture** (Deterministic Machine Learning + Generative AI) and strict **Role-Based Access Control (RBAC)**, ArogyaAI provides a secure, end-to-end ecosystem for both patients and medical practitioners.

---

## 🎯 The Problem & Solution
Traditional Ayurvedic diagnostics rely heavily on practitioner intuition, while modern medical AI models act as "black boxes" that ignore holistic factors like Doshas (Prakriti) and seasonality. Furthermore, exposing raw, low-confidence ML predictions directly to patients poses a severe ethical and psychological risk.

**ArogyaAI solves this by:**
1. Combining mathematical Random Forest predictions with Generative LLM contextual reasoning.
2. Utilizing Explainable AI (XAI) so doctors can see *why* the AI made its decision.
3. Implementing strict Clinical Safety Guardrails that mask low-confidence predictions to prevent patient panic.

---

## ✨ Key Features

### 🔐 Role-Based Architecture (Multi-Tenant)
* **Practitioner Portal:** Doctors have a comprehensive dashboard to run AI diagnostics, view clinic-wide metrics, and manage patient records.
* **Patient Portal:** Patients have a soothing, non-intimidating dashboard to log daily symptoms (Health Diaries) and view safe, actionable Ayurvedic protocols prescribed by their doctor.
* **Clinic ID Siloing:** Data is strictly isolated. Patients link their accounts to a specific doctor using a unique 6-character `Clinic ID`, ensuring secure, HIPAA-compliant-style data routing.

### 🧠 Dual-Engine AI System
* **Engine 1 (Deterministic):** A Python/Flask backend running a trained Random Forest model. It analyzes symptom strings and outputs a disease probability and confidence score.
* **Engine 2 (Generative):** Google Gemini 2.5 LLM analyzes the patient's Dosha, age, gender, and the ML prediction to generate a holistic, personalized Ayurvedic protocol (Herbs & Lifestyle).

### 🛡️ Clinical Safety Guardrails & Ethics
* **Explainable AI (XAI):** Doctors are provided with an "AI X-Ray" showing the exact mathematical weight of each symptom that led to the ML prediction.
* **Confidence Thresholding:** If the AI confidence falls below 35%, the system automatically flags the result as "Inconclusive Data" and warns the doctor, preventing misdiagnosis from vague inputs.
* **Patient View Filtering:** Raw, Western disease labels (e.g., "Typhoid", "AIDS") are masked on the Patient Dashboard. Instead, patients see comforting, actionable advice (e.g., "Personalized Vata Balancing Protocol").

---

## 🚀 Future Roadmap (What's Next?)
ArogyaAI is built to scale. Future iterations of this platform will focus on continuous learning and broader accessibility:

1. **Continuous AI Learning (Feedback Loop):** Allow doctors to "Accept" or "Correct" the AI's diagnosis. Corrected inputs will be saved to a verified database collection to continuously retrain and fine-tune the Random Forest model for specific clinic demographics.
2. **Wearable IoT Integration:** Connect with smartwatches (Apple Watch / Fitbit APIs) to automatically pull real-time vitals (heart rate, sleep data) directly into the Patient's Health Diary.
3. **Multilingual Support for Rural Access:** Integrate translation APIs so patients in rural India can log symptoms in regional languages (Hindi, Marathi, Tamil) while the doctor reads the standardized English output.
4. **Telemedicine & Appointment Booking:** Add a scheduling system where doctors can trigger video calls directly from a patient's high-risk health diary entry.

---

## 🛠️ System Architecture & Tech Stack

**Frontend:**
* React.js (Vite)
* Tailwind CSS (Styling)
* Framer Motion (Fluid Animations)
* Lucide React (Iconography)

**Backend & Cloud:**
* Google Firebase Authentication (Email/Password & Single Sign-On via Google)
* Google Firestore (NoSQL Database for Users, Patient Records, and Diaries)
* Python / Flask (ML API Server)

**Artificial Intelligence:**
* Scikit-Learn (Random Forest Classifier)
* Google Gemini 2.5 Pro API (Generative LLM)

---

## ⚙️ How to Run the Project Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/arogya-ai.git](https://github.com/yourusername/arogya-ai.git)
cd arogya-ai
2. Start the Python AI Server
Ensure you have Python installed, then navigate to the backend directory:

Bash
cd backend
pip install -r requirements.txt
python app.py
The Flask server will start running on http://127.0.0.1:5000.

3. Start the React Frontend
Open a new terminal window, navigate to the frontend directory, and install the Node dependencies:

Bash
cd frontend
npm install
npm run dev
The React app will start running on http://localhost:5173.

👨‍💻 Academic Integrity & Acknowledgements
This project was developed to demonstrate full-stack software engineering, ethical AI implementation, and modern cloud database architecture.

Disclaimer: ArogyaAI is a prototype Clinical Decision Support System. It is designed to assist, not replace, licensed medical professionals.
