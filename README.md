<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3c72,100:2a5298&height=180&section=header&text=Car%20Price%20Prediction&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=35"/>
</p>


**AutoPredict Pro** is an end-to-end **Machine Learning web application** that predicts the **market value of used cars** based on technical specifications, usage details, and seller information.  
The system delivers **accurate, data-driven price estimates** through a **professional Streamlit interface**.

---

🔗 **Live Demo**:  
👉https://carpricepredictionusingmachinelearning-ee9uxjfwakgu4puz5wdjcj.streamlit.app/


## 📌 Problem Statement

Pricing used cars manually is challenging due to multiple influencing factors such as:

- 📅 Manufacturing year & depreciation  
- 🛣️ Mileage and vehicle usage  
- ⚙️ Engine capacity & power output  
- ⛽ Fuel type & transmission mode  
- 👤 Ownership history & brand value  

These complexities often lead to **inconsistent pricing** and **poor decision-making**.  
✨ **AutoPredict Pro** solves this using **AI-driven models** to provide **fair and reliable car price predictions**.

---

## 🏗️ System Architecture

The application follows a structured pipeline to ensure data integrity and prediction accuracy:

1. 🧾 **User Input** – Features collected via Streamlit UI  
2. ✅ **Data Validation** – Logical consistency checks  
3. 🔄 **Preprocessing** – One-Hot Encoding & Feature Scaling (saved pipeline)  
4. 🧠 **Model Inference** – Random Forest or ANN prediction  
5. 💰 **Output** – Price inverse-transformed and displayed in **₹ INR**

---

## 🧠 Models Used

### 1️⃣ Random Forest Regressor (Machine Learning 🌲)
- Handles non-linear relationships effectively  
- Robust to outliers and reduces overfitting  
- ✅ **Best suited for:** General market trends & mid-range vehicles  

### 2️⃣ Artificial Neural Network (Deep Learning 🤖)
- Multi-layer dense neural network (TensorFlow/Keras)  
- Learns complex hidden patterns in high-dimensional data  
- ✅ **Best suited for:** Premium and modern vehicles  

---

## 🛠️ Tech Stack

### 🎨 Frontend & Deployment
- ⚡ **Streamlit** – Interactive web dashboard  
- 🎨 **HTML & CSS** – Custom glass-morphism styling  
- 🎞️ **Lottie Animations** – Smooth UI animations  

### ⚙️ Backend & Machine Learning
- 🐍 **Python** – Core programming language  
- 🌳 **Scikit-learn** – Random Forest & preprocessing pipeline  
- 🧠 **TensorFlow / Keras** – Deep Learning ANN  
- 📊 **Pandas & NumPy** – Data manipulation & numerical computation  

---

## 🔍 Features

### ✅ Core Features
- 🔁 **Dual-Model Prediction** – Compare ML vs DL results in real time  
- 🇮🇳 **INR Formatted Output** – Prices displayed in ₹ for Indian market  
- 🎨 **Professional UI** – Clean, modern & responsive Streamlit design  

### 📊 Market Analysis & Explainable AI (XAI)
- 📌 **Feature Importance** – Impact of Year, Power, KM Driven, etc.  
- 📉 **Depreciation Trends** – How vehicle value decreases over time  
- 📈 **Market Demand Insights** – Fuel & transmission popularity analysis  

---

## 🧪 Input Parameters

| 🔢 Feature | 📝 Description |
|-----------|---------------|
| 🚘 Brand | Car manufacturer (Maruti, BMW, Hyundai, etc.) |
| 📅 Year | Manufacturing year |
| ⛽ Fuel Type | Petrol / Diesel / CNG / LPG |
| 🔧 Transmission | Manual / Automatic |
| 🛣️ KM Driven | Total distance covered |
| ⚙️ Engine | Engine capacity (CC) |
| ⚡ Max Power | Power output (BHP) |
| 👤 Owner | Number of previous owners |
| 🏪 Seller Type | Individual or Dealer |

---

## 🔮 Future Enhancements

- 📍 **Location Intelligence** – City/RTO-based price adjustments  
- 📸 **Computer Vision** – Damage detection from car images  
- 📡 **Live Market Data** – Real-time price updates via APIs  
- 🏢 **B2B Dashboard** – Bulk valuation tools for dealerships  

---

## 👨‍💻 Author

**Lomada Siva Gangi Reddy**  
🎓 B.Tech CSE (Data Science), RGMCET (2021–2025)  
🎯 Aspiring Data Analyst | Python • SQL • Power BI • Data Science  
📍 Open to **Internships & Job Opportunities**
## 📌 Project Workflow

```mermaid
flowchart LR
    A[Importing Libraries] --> B[Loading Car Price Dataset]
    B --> C[Data Cleaning & Feature Engineering]
    C --> D[Preprocessing: Encoding & Scaling]
    D --> E[Model Training: Random Forest / ANN]
    E --> F[Model Evaluation & Selection]
    F --> G[Prediction Output]
    G --> H[Deployment: Streamlit Web App]

    %% Styles
    style A fill:#FFD54F,stroke:#F57F17,stroke-width:2px,color:#000;
    style B fill:#4FC3F7,stroke:#0277BD,stroke-width:2px,color:#fff;
    style C fill:#AED581,stroke:#33691E,stroke-width:2px,color:#000;
    style D fill:#FFCC80,stroke:#EF6C00,stroke-width:2px,color:#000;
    style E fill:#BA68C8,stroke:#4A148C,stroke-width:2px,color:#fff;
    style F fill:#81D4FA,stroke:#01579B,stroke-width:2px,color:#000;
    style G fill:#FF8A65,stroke:#BF360C,stroke-width:2px,color:#fff;
    style H fill:#90CAF9,stroke:#0D47A1,stroke-width:2px,color:#000;
```

## 👨‍💻 Author  

**Madhu Babu Mitta**  
- 🎓 B.Tech IT, TEC (2021–2025)  
- 💡 Interests: Python | Machine Learning | Data Science  
- 📍 Open to **Internships & Job Offers**

 **Contact Me**:  

- 📧 **Email**: madhubabumitta@gmail.com  
- 📞 **Phone**: 7013710799 
- 💼 [LinkedIn](www.linkedin.com/in/madhu-babu-mittta)  🌐 [GitHub](https://github.com/Madhu-Babu-26)  

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3c72,100:2a5298&height=120&section=footer"/>
</p>
