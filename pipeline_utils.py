import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

NEGATIONS = {"not","no","never","nor","n't"}
STOP_WORDS = set("""a an the and or but if while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such only own same so than too very can will just should now i me my we our you your he she it they them this that these those is am are was were be been being have has had do does did as""".split()) - NEGATIONS

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = re.findall(r'[a-z]+', text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t)>1]
    return ' '.join(tokens)

def normalize_rating(value):
    # Converts values like "1", "1.0", "5 stars", "low"/"high" into canonical numeric ratings.
    if pd.isna(value): return None
    s=str(value).strip().lower()
    aliases={'very low':1,'low':2,'neutral':3,'medium':3,'moderate':3,'high':4,'very high':5}
    if s in aliases: return aliases[s]
    m=re.search(r'\b([1-5])(?:\.0)?\b',s)
    return int(m.group(1)) if m else None

def make_model():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=25000, min_df=2)),
        ('classifier', LogisticRegression(max_iter=1200, class_weight='balanced', random_state=42))
    ])

def prepare_data(data):
    data=data[['Review','Rating']].copy()
    data['Rating']=data['Rating'].apply(normalize_rating)
    data['Review']=data['Review'].fillna('').astype(str)
    data=data[data['Rating'].notna()]
    data=data[data['Review'].str.strip().ne('')]
    data=data.drop_duplicates(subset=['Review','Rating'])
    data['Sentiment']=data['Rating'].apply(lambda r: 'Positive' if r>=4 else ('Negative' if r<=2 else 'Neutral'))
    data=data[data['Sentiment']!='Neutral']
    data['Clean_Review']=data['Review'].apply(clean_text)
    data=data[data['Clean_Review'].str.strip().ne('')]
    return data.reset_index(drop=True)
