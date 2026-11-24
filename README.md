# Customer Reviews Sentiment Analysis

A reproducible pipeline and Streamlit interface for sentiment analysis on the supplied Starbucks reviews dataset.

## Data handling decisions
- Retain only `Review` and `Rating` as requested.
- Normalize ratings into 1–5 numeric values, including common textual variants.
- Remove missing/unparseable ratings.
- Ratings 1–2 => Negative; 4–5 => Positive; rating 3 => Neutral and excluded from binary classification.
- Remove duplicate review/rating pairs and empty text.
- Clean text while preserving negation terms (`not`, `no`, `never`).
- Compare TF-IDF + Multinomial Naive Bayes and TF-IDF + Logistic Regression.
- Save the Logistic Regression model as a joblib artifact, ready for Streamlit.

## Run
```bash
pip install -r requirements.txt
python train_pipeline.py
streamlit run streamlit_app.py
```
The dataset file `reviews_data.csv` should be in the project root. Training creates `cleaned_reviews.csv`, `artifacts/sentiment_model.joblib` and metrics JSON.

## GitHub
Create a repository, then:
```bash
git init
git add .
git commit -m "Build sentiment analysis pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sentiment-analysis-reviews.git
git push -u origin main
```
