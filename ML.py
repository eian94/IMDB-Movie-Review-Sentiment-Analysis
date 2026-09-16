import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import joblib
import streamlit
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
set_stopword=set(stopwords.words('english'))
df=pd.read_csv('C:\\Users\\ASUS\\Desktop\\IMDB Dataset.csv')
df=df.drop_duplicates()
def clean_txt(text):
    x=re.sub(r"<.*?>|[.,_,\",-,!,\',:,),(,-]"," ",text)
    x=x.lower()
    words=x.split()
    filter_words=[]
    for word in words:
        if word not in set_stopword:
            filter_words.append(word)
    x=" ".join(filter_words)

    return x
#print(clean_txt(df['review'][1]))
df['review']=df['review'].apply(clean_txt)
#print(df['review'].head())

df['Label']=df['sentiment'].map({'positive':1,'negative':0})



X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['Label'], test_size=0.2, random_state=42
)

vectorizer=TfidfVectorizer()
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)

logestic=LogisticRegression()
multino=MultinomialNB()
logestic.fit(X_train_vec,y_train)
y_pred=logestic.predict(X_test_vec)
#print(accuracy_score(y_test,y_pred))
#print(classification_report(y_test,y_pred))
#print(confusion_matrix(y_test,y_pred))
multino.fit(X_train_vec,y_train)
y_pred_nomi=multino.predict(X_test_vec)
print(accuracy_score(y_test,y_pred_nomi))
joblib.dump(logestic,'sentiment_model.pkl')
joblib.dump(vectorizer,'tfidf_vectorizer.pkl')
import os
#print(os.getcwd())