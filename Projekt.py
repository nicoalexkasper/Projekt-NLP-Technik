import pandas as pd 
from autocorrect import Speller
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import contractions                                     #contractions.fix(<String>)
import nltk
from gensim.models import LdaModel
from gensim.models import LsiModel
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


vectorizerBoW = CountVectorizer(vocabulary=vocabularyList)                         #Nutzung von erzeugtem Vokabular

bowData = vectorizerBoW.fit_transform(reviews)                                     
bowDataDF = pd.DataFrame(bowData.toarray(), columns=vectorizerBoW.get_feature_names_out())

vectorizerTFIDF = TfidfVectorizer(vocabulary=vocabularyList, min_df=1)               #Nutzung von erzeugtem Vokabular + Vokabular muss in mindestens einem Dokument vorkommen
tfidfData = vectorizerTFIDF.fit_transform(vocabularyList)
tfidfDataDF = pd.DataFrame(tfidfData.toarray(), columns = vectorizerTFIDF.get_feature_names_out())


print("BoW Data:")
print(bowDataDF)
print("\n" + "TF-IDF Data:")
print(tfidfDataDF)





corpusGensim = gensim.matutils.Sparse2Corpus(bowData, documents_columns=False)     #BoW von SciKit zu einem Corpus für Gensim
dictionary = gensim.corpora.Dictionary.from_corpus(corpusGensim, id2word=dict((id, word) for word, id in vectorizerBoW.vocabulary_.items())) #Umwandlung zu einem dictionary für id2word

amountOfTopics = 50                                                                 #Auswahl Menge von Themen

lsaModel = LsiModel(corpus=corpusGensim, id2word=dictionary, num_topics=amountOfTopics)
ldaModel = LdaModel(corpus=corpusGensim, id2word=dictionary, num_topics=amountOfTopics)

print( "\n" + "LSA Output:")
print(lsaModel.print_topics())

print( "\n" + "LDA Output:")
print(ldaModel.print_topics())

input()