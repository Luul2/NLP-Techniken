# NLP-Techniken

Dieses Projekt beschäftigt sich mit der Analyse eines Datensatzes von Disneyland-Bewertungen mithilfe verschiedener Methoden des Natural Language Processing (NLP). Nach der Datenvorverarbeitung werden die Texte mithilfe von Bag-of-Words (BoW) und  Term Frequency-Inverse Document Frequency (TF-IDF) in numerische Darstellungen überführt. Um die häufigsten Themen zu erfassen, gibt es zum einen die Latent Semantic Analysis (LSA), welche die Beziehung zwischen Dokumenten und Themen identifiziert. Andererseits existiert die Latent Dirichlet Allocation (LDA), welche die Dirichlet-Verteilung nutzt, um die Themen in den Dokumenten und die Wörter innerhalb der Themen zu modellieren. Zur Bestimmung der optimalen Themenanzahl muss der Coherence Score berechnet werden. Abschließend werden die wichtigsten Begriffe und Ergebnisse mithilfe von Wordclouds visuell dargestellt.

Ziel des Projekts ist es, wiederkehrende Begriffe, zentrale Themen und Stimmungen in den Besucherrezensionen zu identifizieren und die daraus gewonnenen Erkenntnisse zu analysieren und zu bewerten.

## Inhaltsverzeichnis
- [Konzeptionsphase](#Konzeptionsphase)
- [Datenvorverarbeitung](#Datenvorverarbeitung)
- [Vektorisierung](#Vektorisierung)
- [BoW](##BoW)
- [TF-IDF](##TF-IDF)
- [Berechnung des Coherence Scores](#Berechnung-des-Coherence-Scores)
- [Themenmodellierung (LSA & LDA)](#Themenmodellierung)
- [Visualisierung der Ergebnisse](#Visualisierung)
- [Herausforderungen](#herausforderung)

## Konzeptionsphase
Zunächst wird der Datensatz mit den Disneyland-Bewertungen als CSV-Datei über Kaggle bezogen und ein erster Überblick über die enthaltenen Daten gewonnen.

Eine Besucherbewertung über das Disneyland lautet beispielsweise: „This place is HUGE! Definately need more than one day. We had 3 children aged 11, 9 & 6“. Anhand dieses Beispiels wird die Notwendigkeit der Datenvorverarbeitung zur Textbereinigung verdeutlicht, welche wie folgt aussieht:
 - Konvertierung des Textes in Kleinbuchstaben „this place is huge …“
 - Entfernung von Sonderzeichen, Zahlen (z. B „!“, „3“) und Stoppwörtern (z. B. „we“)
 - Extrahierung von Einzelwörtern
 - Durchführung der Lemmatisierung (z. B. „is“ wird zu „be“)

Für die Umsetzung der einzelnen Verarbeitungsschritte werden folgende Python-Bibliotheken eingesetzt:
 - Einlesen der Daten in Python ⇨ pandas
 - Textbereinigung ⇨ nltk
 - Datenkonvertierung und Themenextraktion ⇨ scikit-learn
 - Kohärenzberechnung ⇨ gensim
 - Visualisierung der häufigsten Wörter ⇨ wordcloud, matplotlib, pillow, numpy

Zu Beginn werden in PyCharm die benötigten Bibliotheken installiert und in das Python-Projekt eingebunden. Anschließend wurden die Disneyland-Bewertungen mithilfe von `read_csv()` aus pandas eingelesen und für die weitere Verarbeitung bereitgestellt.

## Datenvorverarbeitung
Mithilfe der Funktion `preprocess_text` wird die Textvorverarbeitung durchgeführt. In dieser wird der Text in einzelne Wörter, sogenannte Tokens, aufgeteilt und in Kleinbuchstaben umgewandelt. Zudem erfolgt eine Entfernung der Stoppwörter mithilfe der englischen Stoppwortliste aus der Bibliothek von `nltk`. Zusätzlich werden Sonderzeichen und Zahlen herausgefiltert, sodass für die weitere Analyse ausschließlich alphabetische Wörter erhalten bleiben. Außerdem findet die Lemmatisierung statt, bei der die Wörter in ihre kanonische Grundform umgewandelt werden. Abschließend gibt die Funktion den vorverarbeiteten Text mit return `' '.join(words)` wieder als zusammenhängenden String zurück.

Nach der Erstellung der Funktion wird `preprocess_text` auf die Spalte Review_Text angewendet. Die bereinigten Daten werden anschließend in einer neuen CSV-Datei gespeichert und für die weiteren Analyseschritte bereitgestellt.

Überprüfung der bereinigten Daten durch eine Gegenüberstellung mit den Originaldaten:

![NLP](bilder/1.png)

## Vektorisierung 
Um die vorverarbeiteten Texte in eine numerische Darstellung zu überführen, erfolgt im nächsten Schritt die Vektorisierung mit Bag-of-Words (BoW) und TF-IDF. Hierfür werden die Klassen `CountVectorizer` und `TfidfVectorizer` aus der `scikit-learn`-Bibliothek verwendet. Bei der Vektorisierung werden sowohl Unigramme als auch Bigramme berücksichtigt. Zusätzlich wird die Anzahl auf 1000 Wörter beziehungsweise n-Gramme begrenzt. Das Ergebnis wird in einem DataFrame mit `pandas` gespeichert und ausgegeben. 

### BoW
Bei der BoW-Vektorisierung ist zu erkennen, dass das Wort „absolutely“ in der ersten Kundenbewertung  den Wert 1 erhält. Dieser Wert gibt an, dass „absolutely“ genau einmal in der entsprechenden Rezension vorkommt.

![NLP](bilder/2.png)

### TF-IDF
Anders verhält es sich bei der TF-IDF-Vektorisierung, bei der das Wort „absolutely“ einen Wert von 0.216205 erhält. Dieser Wert ergibt sich aus der Häufigkeit des Wortes innerhalb der jeweiligen Rezension sowie aus seiner Häufigkeit im gesamten Korpus. Im Gegensatz zu BoW berücksichtigt TF-IDF somit nicht nur, wie häufig ein Wort in einer Rezension vorkommt, sondern auch, wie häufig es in den übrigen Rezensionen vertreten ist.

![NLP](bilder/3.png)

## Berechnung-des-Coherence-Scores
