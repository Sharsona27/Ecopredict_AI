# 🌱 EcoPredict AI

EcoPredict AI is a smart web-based application that helps users predict **energy consumption**, **carbon emissions**, and **electricity costs** using Machine Learning and an AI-powered chatbot.

---

## 🚀 Features

- 🔋 Energy Consumption Prediction  
- 🌍 Carbon Emission Calculation  
- 💰 Electricity Cost Estimation  
- 🤖 AI Chatbot (EcoBot) for energy-saving tips  
- 📊 Interactive Dashboard  
- 🔐 User Authentication System (Login/Signup)  
- 🧠 Machine Learning Model Integration  

---

## 🧠 Tech Stack

**Frontend:**
- HTML
- CSS

**Backend:**
- Flask (Python)

**Database:**
- SQLite

**Machine Learning:**
- Scikit-learn
- NumPy
- Joblib

**AI Integration:**
- Hugging Face Inference API

**Other Tools:**
- dotenv
- Flask-CORS

---

## 📂 Project Structure
```bash
EcoPredict_AI/
│── app.py
│── train_model.py
│── improved_model_training.py
│
├── dataset/
│ └── energydata_complete.csv
│
├── model/ (excluded from GitHub)
│ └── *.pkl
│
├── templates/
├── static/
├── instance/
│
└── README.md


---

## ⚙️ Installation & Setup

```bash
1️⃣ Clone the Repository
git clone https://github.com/Sharsona27/EcoPredict_AI.git
cd EcoPredict_AI
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Setup Environment Variables
Create a .env file in the root directory:

HF_API_KEY=your_huggingface_api_key
FLASK_SECRET_KEY=your_secret_key
▶️ Run the Application
python app.py
Open in browser:
http://127.0.0.1:5000/

📊 Dataset
Original dataset and processed dataset are used for training the model
Due to size constraints, datasets are shared via Google Drive
👉 Dataset Link: (Add your Drive link here)

🤖 AI Chatbot (EcoBot)
Powered by Hugging Face Inference API
Provides:
Energy-saving tips
General assistance
Guidance for predictions

🧠 Model Files
Large .pkl model files are not uploaded to GitHub
Download from Google Drive
  Model Files Link: (Add your Drive link here)

🌐 Deployment
   Live Project Link: (Add deployed link here)

🔮 Future Improvements
Improve prediction accuracy
Add real-time data integration
Enhance UI/UX
Add mobile support

📄 License
This project is licensed under the MIT License.
