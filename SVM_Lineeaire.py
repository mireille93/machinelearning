# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 01:56:19 2019

@author: Mireille/Henri
"""

# Methode de SVM lineaire

# importing the Dataset

import pandas as pd

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

# telechhargement du jeu de donnnees en format csv les label separe des messages par des esoaces
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])


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

 
# Training model using SVM classifier

from sklearn.svm import SVC
#recuperation de l'objet SVM lineaire
spam_detect_model = SVC(kernel='linear', random_state = 0)

spam_detect_model.fit(X_train, y_train)

#prediction sur les jeux de test
y_pred=spam_detect_model.predict(X_test)

print("**********************************************************************")

def pourcentage_reussite(y_pred, y_test):
    count = 0
    pourcentage = 0
    vecteur = []
    for i in range(len(y_pred)):
        if(y_pred[i]==y_test[i]):
            count = count+1
        else:
            vecteur.append(y_test[i])
            
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
print("***visualisation des faux positif et des vrai negatif en diagramme*****")

import matplotlib.pyplot as plt
import numpy as np
x = [1,2]
y = [confusion_m[1,0],confusion_m[0,1]]
plt.bar(x,y)

plt.xlabel('vrai négatif                         /             faux Positif')
plt.ylabel('nombres de mail correspondant')
plt.title('diagramme de comparaison matrice confusion')
