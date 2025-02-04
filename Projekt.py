import pandas as pd 
from autocorrect import Speller
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import contractions
import nltk
from gensim.models import LdaModel
from gensim.models import LsiModel
from gensim.models import CoherenceModel
import gensim
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

names = ["xfinity"]                                                                                                                                                         #Namen die nicht korregiert werden sollten; Hier nur Xfinity; Jedoch durch Array erweiterbar

def correctSpellingMistakes(sentence):                                                                                                                                      #Verbessern der Rechtschreibfehler
    spell = Speller("en")
    words = sentence.split()
    tempSentence = ""
    for word in words:
        if word == names:
            tempSentence += spell(word) + " "
        else:
            tempSentence += word + " "
    return tempSentence

def completeContractions(sentence):                                                                                                                                         #Vervollständigung von Kontraktionen ("They're" -> "They are")
    return contractions.fix(sentence)

def removeSpecialChars(sentence):                                                                                                                                           #Entfernung von speziellen Zeichen ("e-mail" -> "email")
        tempSentence = ""
        for chars in sentence:
            if chars.isalpha():
                tempSentence += chars
            elif chars == "-":
                break
            else:
                tempSentence +=" "
        return tempSentence


vocabularyList = []
tokenizedList = []

reviewsdf = pd.read_csv("Comcast.csv")                                                                                                                                      #CSV Datei zu Dateframe
reviews = reviewsdf["Customer Complaint"].tolist()                                                                                                                          #Filtern der Spalte Customer Complaint aus dem Dataframe, Ergebnis sind Reviews 


for element in reviews:
    element = element.lower()
    element = correctSpellingMistakes(element)                                                                                                                             #Aufruf von Methode correctSpellingMistakes
    element = completeContractions(element)                                                                                                                                #Aufruf von Methode completeContractions
    element = removeSpecialChars(element)                                                                                                                                  #Aufruf von Methode removeSpecialChars

    lemmatizer =  nltk.stem.WordNetLemmatizer()
    tokens = nltk.tokenize.word_tokenize(element)                                                                                                                           #Tokenize von Review
    stopWords = set(stopwords.words('english'))                                                                                                                             #Nutzung Englische Stoppwörter
    stopWords.remove("no")                                                                                                                                                  #Entfernung aus der Stoppwörter Liste no
    stopWords.remove("not")                                                                                                                                                 #Entfernung aus der Stoppwörter Liste not
    filteredSentence = [lemmatizer.lemmatize(word) for word in tokens if not word in stopWords]                                                                             #Entfernung von Stoppwörtern in der Review & lemmatisierung der Wörter
    for word in filteredSentence:
        tokenizedList.append(filteredSentence)                                                                                                                              #Tokenized Wörter werden zur Liste für Berechnung Coherence Score gespeichet
        if word not in vocabularyList:                                                                                                                                      #Sofern das Wort nicht in der Vokabel-Liste vorkommt, wird dies hinzugefügt
            vocabularyList.append(word)


vectorizerBoW = CountVectorizer(vocabulary=vocabularyList)                                                                                                                  #Nutzung von erzeugtem Vokabular

bowData = vectorizerBoW.fit_transform(reviews)                                                                                                                              #Erstellung BoW
bowDataDF = pd.DataFrame(bowData.toarray(), columns=vectorizerBoW.get_feature_names_out())                                                                                  #Erstellung DataFrame für BoW

vectorizerTFIDF = TfidfVectorizer(vocabulary=vocabularyList, min_df=1)                                                                                                      #Nutzung von erzeugtem Vokabular + Vokabular muss in mindestens einem Dokument vorkommen
tfidfData = vectorizerTFIDF.fit_transform(vocabularyList)                                                                                                                   #Erstellung TF-IDF
tfidfDataDF = pd.DataFrame(tfidfData.toarray(), columns = vectorizerTFIDF.get_feature_names_out())                                                                          #Erstellung DataFrame für BoW


print("BoW Data:")
print(bowDataDF)                                                                                                                                                            #Darstellung BoW-Vektor
print("\n" + "TF-IDF Data:")
print(tfidfDataDF)                                                                                                                                                          #Darstellung TF-IDF-Vektor





corpusGensim = gensim.matutils.Sparse2Corpus(bowData, documents_columns=False)                                                                                              #BoW von SciKit zu einem Corpus für Gensim
dictionary = gensim.corpora.Dictionary.from_corpus(corpusGensim, id2word=dict((id, word) for word, id in vectorizerBoW.vocabulary_.items()))                                #Umwandlung zu einem dictionary für id2word


amountOfTopics = 10                                                                                                                                                         #Auswahl Anzahl von Themen

lsaModel = LsiModel(corpus=corpusGensim, id2word=dictionary, num_topics=amountOfTopics)                                                                                     #Erstellung LSA-Modell
ldaModel = LdaModel(corpus=corpusGensim, id2word=dictionary, num_topics=amountOfTopics)                                                                                     #Erstellung LDA-Modell

lsaTopics = [[word for word, prob in topic] for topicid, topic in lsaModel.show_topics(formatted=False)]                                                                    #Output von Themen des LSA-Modells
ldaTopics = [[word for word, prob in topic] for topicid, topic in ldaModel.show_topics(formatted=False)]                                                                    #Output von Themen des LDA-Modells



coherenceModelLSA = CoherenceModel(model=lsaModel, coherence='c_v', texts=tokenizedList, corpus=corpusGensim, dictionary=dictionary, topics=lsaTopics, processes=1)         #Erstellung Coherence-Modell von LSA-Modell
coherenceScoreLSA = coherenceModelLSA.get_coherence()                                                                                                                       #Coherence-Wert von LSA-Modell

coherenceModelLDA = CoherenceModel(model=ldaModel, coherence='c_v', texts=tokenizedList, corpus=corpusGensim, dictionary=dictionary, topics=ldaTopics, processes=1)         #Erstellung Coherence-Modell von LDA-Modell
coherenceScoreLDA = coherenceModelLDA.get_coherence()                                                                                                                       #Coherence-Wert von LDA-Modell


topicsLSA = [ str(t) + "\n" for t in lsaTopics ]                                                                                                                            #Sortieren von Themen des LSA-Modells
topicsLDA = [ str(t) + "\n" for t in ldaTopics ]                                                                                                                            #Sortieren von Themen des LDA-Modells
dataTopicScoreLSA = pd.DataFrame( data=zip(topicsLSA, coherenceModelLSA.get_coherence_per_topic()), columns=['Topic', 'Coherence'] )                                        #Erstellung DataFrame für die Darstellung Themen & Coherence für das Thema (LSA)
dataTopicScoreLDA = pd.DataFrame( data=zip(topicsLDA, coherenceModelLDA.get_coherence_per_topic()), columns=['Topic', 'Coherence'] )                                        #Erstellung DataFrame für die Darstellung Themen & Coherence für das Thema (LDA)


print( "\n" + "LSA Output:")                                                                                                                                                
print(dataTopicScoreLSA)                                                                                                                                                    #Output DataFrame Thema & Coherence des LSA-Modells
print("\n***" + " Total Coherence Score: " + str(coherenceScoreLSA) + " ***\n")                                                                                             #Output Total Coherence Score für das LSA-Modell

print( "\n" + "LDA Output:")
print(dataTopicScoreLDA)                                                                                                                                                    #Output DataFrame Thema & Coherence des LDA-Modells
print("\n***" + " Total Coherence Score: " + str(coherenceScoreLDA) + " ***\n")                                                                                             #Output Total Coherence Score für das LDA-Modell




input()