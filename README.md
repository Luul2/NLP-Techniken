## NLP-Techniken

Dieses Projekt beschäftigt sich mit der Analyse eines Datensatzes von Disneyland-Bewertungen mithilfe verschiedener Methoden des Natural Language Processing (NLP). Nach der Datenvorverarbeitung werden die Texte vektorisiert und mithilfe der Themenmodellierung (LSA und LDA) analysiert. Abschließend werden die wichtigsten Begriffe und Ergebnisse durch Wordclouds und weitere Visualisierungen anschaulich dargestellt.

Ziel des Projekts ist es, wiederkehrende Begriffe, zentrale Themen und Stimmungen in den Besucherrezensionen zu identifizieren und die gewonnenen Erkenntnisse zu bewerten.

## Inhaltsverzeichnis
- [Datenvorverarbeitung](#Datenvorverarbeitung)
- [Vektorisierung (BoW & TF‑IDF)](#Vektorisierung))
- [Themenmodellierung (LSA & LDA)](#Themenmodellierung))
- [Berechnung des Coherence Scores](#Berechnung-des-Coherence-Scores)
- [Visualisierung der Ergebnisse](#Visualisierung)
- [Herausforderungen](#herausforderung)

## Datenvorverarbeitung
Zu Beginn wurden in PyCharm die benötigten Bibliotheken wie pandas, nltk, scikit-learn und gensim installiert und eingebunden. Anschließend wurden die Disneyland-Bewertungen mithilfe von `read_csv()` aus pandas eingelesen.

Im nächsten Schritt wurde die Funktion `preprocess_text` erstellt, in der die Textvorverarbeitung durchgeführt wurde. Diese umfasste die Tokenisierung, die Umwandlung in Kleinbuchstaben, die Entfernung von Stoppwörtern mithilfe der nltk-Stoppwortliste sowie das Entfernen von Sonderzeichen und Zahlen. Zusätzlich wurde eine Lemmatisierung durchgeführt, bei der Wörter in ihre Grundform überführt werden. Abschließend wurden die verarbeiteten Tokens wieder zu einem String zusammengeführt.

Überprüfung der bereinigten Daten durch eine Gegenüberstellung mit den Originaldaten:

![NLP Bild1](Bilder/1.png)

## Vektorisierung 
