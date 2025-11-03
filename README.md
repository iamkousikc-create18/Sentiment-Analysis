💬 Sentiment Analysis using NLP

A machine learning project that analyzes the sentiment of text (Positive/Negative/Neutral) using *Natural Language Processing (NLP)* techniques and *Logistic Regression. The project also features a **Streamlit web app* for real-time sentiment prediction.

---

📌 Project Overview

This project aims to classify user-input text based on its sentiment.  
It processes text using NLP techniques, converts it into numerical form using *TF-IDF, and applies a **Logistic Regression* model for sentiment classification.

✅ User can input any sentence and instantly get the sentiment result.  
✅ Deployed using *Streamlit* for an interactive and user-friendly interface.  

---

🛠 Tech Stack

| Component | Technology Used |
|-----------|------------------|
| Programming | Python |
| NLP | NLTK / SpaCy |
| ML Algorithm | Logistic Regression |
| Feature Extraction | TF-IDF Vectorizer |
| Deployment | Streamlit |
| Libraries | Pandas, NumPy, Scikit-learn |

---

🔄 End-to-End Workflow

1️⃣ Data Collection  
- Collected dataset containing text and corresponding sentiment labels.  
- Examples: movie reviews, customer feedback, tweets, etc.

---
2️⃣ Data Preprocessing (NLP)  
✔ Lowercasing  
✔ Removal of stopwords & punctuation  
✔ Tokenization  
✔ Lemmatization/Stemming  
✔ Text cleaning  

3️⃣ Feature Engineering  
- Converted clean text into numerical features using *TF-IDF (Term Frequency – Inverse Document Frequency)*.

4️⃣ Model Training  
- Applied *Logistic Regression* for sentiment prediction.  
- Trained model saved as model.pkl and vectorizer saved as tfidf_vectorizer.pkl.  

5️⃣ Streamlit Web App  
- Created app.py using Streamlit.  
- User inputs text ➝ model predicts sentiment ➝ result displayed on screen.  
- Simple and interactive UI.

---

## 📁 Project Structure

📦 sentiment-analysis-nlp/ │ ├── Sentiment.ipynb                    # Streamlit web app ├── updatesenti.py                 # Trained Logistic Regression model ├── sentu.pkl       # Saved TF-IDF vectorizer ├──  scaler.pkl          # All dependencies ├── README.md                  # Project documentation └── data/                      # Dataset file ├──  IMDB Dataset.csv

---

🎯 Results

Input Sentence	Predicted Sentiment

"I really love this product!"	✅ Positive
"This is the worst experience ever."	❌ Negative
"It's okay, nothing special."	⚪ Neutral



---

🧠 Key Learnings

✔ Applied end-to-end NLP pipeline
✔ Hands-on with TF-IDF & Logistic Regression
✔ Built and deployed a Streamlit web application
✔ Improved understanding of text classification workflow


---

🚀 Future Enhancements

Add deep learning models (LSTM, BERT)

Include multilingual sentiment detection

Add dataset exploration and better visualizations

Deploy on Render / Hugging Face / Streamlit Cloud



---

👨‍💻 Author
👤 Kousik Chakraborty
📧 Email: www.kousik.c.in@gmail.com
🔗 GitHub Profile: https://github.com/iamkousikc-create18
🔗 Project Repository: https://github.com/iamkousikc-create18/Sentiment-Analysis
