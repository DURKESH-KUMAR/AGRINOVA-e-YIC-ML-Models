# 🌾 AGRINOVA — e-YIC ML Models

AGRINOVA is a machine learning-based project designed to predict agricultural storage conditions, specifically focusing on **onion spoilage prediction** using environmental and sensor data.

This project was developed as part of the **e-YIC (Youth Innovation Challenge)** to provide smart, data-driven insights for improving post-harvest storage and reducing crop loss.

---

## 🚀 Features

- 📊 Predicts spoilage levels based on sensor data  
- 🌡️ Supports environmental parameters (Temperature, Humidity, Gas levels)  
- 🤖 Uses Machine Learning models (Random Forest, Gradient Boosting)  
- 💾 Stores trained models using Pickle (`.pkl`)  
- 🖥️ Includes a Tkinter-based UI for easy interaction  
- 📁 Works with CSV input datasets  

---

## 🧠 Machine Learning Models Used

- Random Forest Regressor  
- Gradient Boosting Regressor  

These models are trained to:
- Predict warning days
- Predict critical spoilage days

---

## 📂 Project Structure

```
AGRINOVA-e-YIC-ML-Models/
│
├── data/                     
├── models/                   
├── ui/                       
├── scripts/                  
├── outputs/                  
│
├── agrinova_model.pkl        
├── main.py                   
├── requirements.txt          
└── README.md                 
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/DURKESH-KUMAR/AGRINOVA-e-YIC-ML-Models.git
cd AGRINOVA-e-YIC-ML-Models
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the application:

```bash
python main.py
```

- Upload your CSV file
- Provide a session name
- Run the prediction pipeline

---

## 📊 Input Data Format

Your CSV should include:

- Temperature  
- Humidity  
- Gas levels  

---

## 📦 Output

- Predicted warning days  
- Predicted critical days  
- Saved model file (`.pkl`)  

---

## 🎯 Objective

To reduce agricultural losses by providing intelligent predictions for storage conditions using machine learning.

---

## 📚 Future Improvements

- Deep learning integration  
- Real-time IoT sensor connectivity  
- Web dashboard deployment  

---

## 👨‍💻 Author

**Durkesh Kumar**

---

## 📜 License

This project is open-source and available under the MIT License.
