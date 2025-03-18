from fastcoref import spacy_component
import spacy


text = 'Alice goes down the rabbit hole. Where she would discover a new reality beyond her expectations.'

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("fastcoref")

doc = nlp(text)
print(doc._.coref_clusters)