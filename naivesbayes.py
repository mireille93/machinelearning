# NAIVES BAYES

# importing the Dataset avec pandas


import pandas as pd
import matplotlib.pyplot as plt

# telechhargement du jeu de donnnees en format csv les label separe des messages par des esoaces
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])

#Pretraitement
import re
import nltk

#stopwords contient des mots courant en anglais les plus utilises
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
#les mots de stopwords sont compares et supprimes du jeu de donnees
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
# Utilisation de 2500 mots les plus frequents du jeu de donnnes
# la matrice X contient une matrice de 5572,2500 de valeur 0/1
# y est contient la transformation des label transformes en (ham=0/spam=1) du jeu de donnees
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train Test Split (partage du jeu de donnees pour les tests, 20% de la taille totale)
# 80% restant pour l'entraienement

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

 

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB

#recuperation de l'objet Multinomial naives bayes
spam_detect_model = MultinomialNB().fit(X_train, y_train)

#faire la predicton des donnees spam et mail ordinaire dans y_pred
y_pred=spam_detect_model.predict(X_test)

#fonction pour calculer le pourcentage de reussite de la methode de 
#classification par rapport aux resltats predits et les tests entraine

def pourcentage_reussite(y_pred, y_test):
    count = 0
    pourcentage = 0
    for i in range(len(y_pred)):
        if(y_pred[i]==y_test[i]):
            count = count+1
        
    pourcentage = count*100/len(y_pred)    
    return pourcentage
print("************************************************************")
print('pourcentage de reuissite est : ',pourcentage_reussite(y_pred, y_test),'\npourcentage d''echec est :',(100-pourcentage_reussite(y_pred, y_test)))
print("************************************************************")

#matrice de confusion des resultats de prediction et de tests
from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)

# accuracy ou indice de resussite des predictions sur les tests
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)


# visualisation des faux positif et des vrai negatif en diagramme

import numpy as np
x = [1,2]
y = [confusion_m[1,0],confusion_m[0,1]]
plt.bar(x,y)

plt.xlabel('vrai négatif                         /             faux Positif')
plt.ylabel('nombres de mail correspondant')
plt.title('diagramme de comparaison matrice confusion')












