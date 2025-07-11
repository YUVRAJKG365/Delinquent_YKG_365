# 📊 Customer Delinquency Prediction System – AI-Powered Risk Assessment & Fairness-Aware Modeling

A production-ready, end-to-end AI system for predicting **customer delinquency**, developed using real-world behavioral, financial, and demographic data. This system integrates **machine learning**, **bias mitigation**, and **fairness auditing** using the **Fairlearn** framework, ensuring responsible AI outcomes in high-stakes financial decisioning.

> 💡 Developed by **Yuvraj Kumar Gond** as part of a real-world applied AI challenge focused on fairness, explainability, and ethical modeling in finance.

---

## 🚀 Key Features

✅ **Interactive Streamlit App** – Real-time delinquency risk prediction & business recommendations  
🔍 **AI Model Pipeline** – Trained with Logistic Regression & Random Forest; uses class balancing (SMOTE/upsampling)  
📈 **Explainable Risk Drivers** – Missed payments, DTI ratio, credit utilization, late fees, employment status  
🧠 **Fairlearn Integration** – Ensures fairness across **employment type**, **location**, and other sensitive features  
📊 **Bias Auditing Dashboard** – Includes Demographic Parity, Equalized Odds, and subgroup metrics  
💡 **Smart Recommendations** – Actionable insights to guide collections and customer support teams  
🔐 **Ethical AI by Design** – Focus on fairness, transparency, and human-centered impact

---

## 🧪 Model Performance

| Metric                 | Value   |
|------------------------|---------|
| Accuracy               | 88.8%   |
| Precision              | 94%   |
| Recall (TPR)           | 91%   |
| Demographic Parity Δ   | 0.237   |
| Equalized Odds Δ       | 0.091   |

✅ **Fairlearn post-processing** significantly reduces bias while maintaining high predictive performance.

---

## 🎯 Business Use Case

This AI solution supports financial institutions like **Geldium** in optimizing their collections and outreach strategies by:

- 🔎 **Identifying high-risk customers** 90 days in advance  
- 🎯 **Targeting support or intervention** before default happens  
- 💸 **Reducing collections cost** by up to 35%  
- ✅ **Improving compliance** with ethical and legal AI standards  
- 🤝 **Building customer trust** through transparency and fairness

---

## 🛠️ Tech Stack

| Component         | Tools Used                             |
|------------------|----------------------------------------|
| Programming       | Python (Pandas, NumPy, scikit-learn)   |
| ML Pipeline       | Logistic Regression, Random Forest     |
| UI Interface      | Streamlit                              |
| Fairness Auditing | Fairlearn                              |
| Balancing Data    | imbalanced-learn                       |
| Persistence       | Joblib                                 |
| Exploration       | Jupyter Notebook                       |

---

## 🤖 Model Ethics & Explainability

This project incorporates responsible AI practices:

- 📊 **Bias detection** on sensitive groups (Employment Type, Location, Age)
- 🧩 **Post-processing with Fairlearn** using Equalized Odds
- 🔍 **Transparent ML predictions** with explainable reasoning
- 📣 **Human-centered design** for non-technical users to understand risk decisions

---

## 📌 Acknowledgment

🧠 **Developed by:** Yuvraj Kumar Gond  
🏢 **Use Case Context:** Applied AI challenge focused on **financial fairness and responsible AI**  
📚 **Frameworks Used:** Fairlearn, Streamlit, Scikit-learn, imbalanced-learn

> If you find this project valuable, consider giving it a ⭐ on GitHub and sharing with your network!
