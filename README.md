# FinGuard: Customer Risk & Churn Intelligence Agent

An end-to-end hybrid AI architecture for predictive customer retention in fintech banking, combining Machine Learning accuracy with Generative AI reasoning.

## 🏗️ Project Architecture
The platform processes data through four production-ready layers:
1. **Data Warehouse Layer:** A relational `MySQL` database consolidating customer demographic and behavioral traits.
2. **Predictive Engine Layer:** A high-performance `Random Forest Classifier` built in Python using `scikit-learn` ($95.5\%$ accuracy) to calculate exact customer churn probabilities.
3. **The Gatekeeper Rule:** A business logic threshold ensuring high-risk profiles ($>75\%$) automatically trigger downstream cognitive action.
4. **Cognitive Layer (AI Agent):** An autonomous system powered by the modern Google GenAI SDK (`gemini-2.5-flash`) that ingests structured customer context to generate internal risk analyses, targeted retention mechanics, and personalized customer outreach emails.

## 🚀 Technical Stack
* **Language:** Python 3.10+
* **Data Layer:** MySQL, SQLAlchemy, Pandas
* **Analytics & ML:** Scikit-Learn, NumPy
* **Generative AI:** google-genai SDK (Gemini 2.5 Flash)
* **Business Intelligence:** Tableau (Executive Dashboarding)

---

## 📦 How to Run Separately
1. Clone the repository.
2. Configure your local database and save your Gemini API key in `api_key.txt`.
3. Install required packages:
   ```bash
   pip install pandas sqlalchemy pymysql scikit-learn google-genai