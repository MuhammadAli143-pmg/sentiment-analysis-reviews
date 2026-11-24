import os, json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from pipeline_utils import prepare_data

DATA_PATH=os.getenv('DATA_PATH','reviews_data.csv')
os.makedirs('artifacts',exist_ok=True)
raw=pd.read_csv(DATA_PATH)
clean=prepare_data(raw)
clean.to_csv('cleaned_reviews.csv',index=False)
X=clean['Clean_Review']; y=clean['Sentiment']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models={
 'MultinomialNB':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),max_features=25000,min_df=2)),('model',MultinomialNB())]),
 'LogisticRegression':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),max_features=25000,min_df=2)),('model',LogisticRegression(max_iter=1200,class_weight='balanced',random_state=42))])
}
results={}
for name,m in models.items():
 m.fit(X_train,y_train); p=m.predict(X_test)
 results[name]={'accuracy':accuracy_score(y_test,p),'precision':precision_score(y_test,p,pos_label='Positive'),'recall':recall_score(y_test,p,pos_label='Positive'),'f1':f1_score(y_test,p,pos_label='Positive'),'confusion_matrix':confusion_matrix(y_test,p,labels=['Negative','Positive']).tolist()}
 print('\\n',name,results[name]);print(classification_report(y_test,p))
best=models['LogisticRegression']
joblib.dump(best,'artifacts/sentiment_model.joblib')
with open('artifacts/metrics.json','w') as f:json.dump(results,f,indent=2)
print('Saved model to artifacts/sentiment_model.joblib')
