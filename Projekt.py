import pandas as pd 
from autocorrect import Speller
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import contractions                                     #contractions.fix(<String>)
import nltk
import gensim
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def lowerElementsInList(list):                                                   #Nimmt Liste und macht diese zu Lowercase
    return [element.lower() for element in list]

def correctSpellingMistakes(list):                                              #Verbessern der Rechtschreibfehler
    spell = Speller("en")
    tempReview = []
    for element in list:
        word = spell(element)
        tempReview.append(word)
    return tempReview

def completeContractions(list):                                                 #Vervollständigung von Kontraktionen ("They're" -> "They are")
    return[contractions.fix(element) for element in list]

def removeSpecialChars(list):
    tempList = []
    for element in list:
        tempSentence = ""
        for chars in element:
            if chars.isalpha():
                tempSentence += chars
            elif chars == "-":
                break
            else:
                tempSentence +=" "
                
        tempList.append(tempSentence)
    return tempList


vocabularyList = []

reviewsdf = pd.read_csv("Comcast.csv")                                          #CSV Datei zu Dateframe
reviews = reviewsdf["Customer Complaint"].tolist()                              #Filtern der Spalte Customer Complaint aus dem Dataframe, Ergebnis sind Reviews 
reviews = lowerElementsInList(reviews)                                          #Aufruf von Methode lowerElementsInList
reviews = correctSpellingMistakes(reviews)                                      #Aufruf von Methode correctSpellingMistakes
reviews = completeContractions(reviews)                                         #Aufruf von Methode completeContractions
reviews = removeSpecialChars(reviews)                                           #Aufruf von Methode completeContractions


for element in reviews:
    lemmatizer =  nltk.stem.WordNetLemmatizer()
    tokens = nltk.tokenize.word_tokenize(element)
    stop_words = set(stopwords.words('english'))
    stop_words.remove("no")
    stop_words.remove("not")
    filteredSentence = [word for word in tokens if not word in stop_words]
    for word in filteredSentence:
        if lemmatizer.lemmatize(word) not in vocabularyList:
            vocabularyList.append(lemmatizer.lemmatize(word))


vectorizer = CountVectorizer(vocabulary=vocabularyList)                         #Nutzung von erzeugtem Vokabular

bowData = vectorizer.fit_transform(reviews)                                     
bowData = pd.DataFrame(bowData.toarray(), columns=vectorizer.get_feature_names_out())

vectorizer = TfidfVectorizer(vocabulary=vocabularyList, min_df=1)
model = vectorizer.fit_transform(vocabularyList)
tfidfData = pd.DataFrame(model.toarray(), columns = vectorizer.get_feature_names_out())


print("BoW Data:")
print(bowData)
print("TF-IDF Data:")
print(bowData)
input()