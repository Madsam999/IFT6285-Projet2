import spacy
from datasets import load_dataset
from tqdm import tqdm
import re

def extract_named_entities(model, corpus):
    for sentence in tqdm(corpus, desc = "Extraction des entitées nommées"):
        doc = model(sentence)

def preprocess_corpus(dataset):

    texts = dataset["train"]["article"]

    regexPattern = r"=\s=\s.+?\s=\s="

    sentences = []
    for text in tqdm(texts, desc = "Pré-traitement du texte"):
        if not text.strip():
            continue
        cleaned_text = re.sub(regexPattern, " ", text)
        preprocessedText = cleaned_text.strip().lower()
        if preprocessedText:
            sentences.append(preprocessedText)
    return sentences

def extractNER_optimized(nlp, corpus):
    results = []
    
    # Utilisation de nlp.pipe() pour le traitement par lots
    # batch_size=50 ou 100 est un bon point de départ.
    # n_process=-1 utilise tous les cœurs disponibles pour le parallélisme.
    
    print("Début de l'extraction parallèle et par lots...")
    
    # Nous itérons sur les objets Doc générés par nlp.pipe()
    # Nous utilisons tqdm sur la sortie de nlp.pipe pour la barre de progression.
    doc_generator = nlp.pipe(
        corpus, 
        batch_size=100, 
        n_process=3
    ) 
    
    for doc in tqdm(doc_generator, desc="Extraction des entités nommées", total=len(corpus)):
        textResults = []
        # La boucle d'extraction des jetons et de leurs types d'entités reste la même
        for token in doc:
            textResults.append((token.text, token.ent_type_ if token.ent_type_ else None))
        results.append(textResults)
        
    return results
        


if __name__ == '__main__':
    # Tout le code qui exécute l'extraction doit être ici
    
    # 1. Chargement des ressources
    ds = load_dataset("Mouwiya/wikitext103-train")
    nlp = spacy.load("en_core_web_lg")

    # 2. Exécution du pipeline
    print("Début du pré-traitement du dataset Wikitext103-train")
    corpus = preprocess_corpus(ds)
    print(f"Pré-traitement terminé. {len(corpus)} documents à traiter.")

    print("Extraction des NER du corpus (Optimisée avec nlp.pipe)")
    results = extractNER_optimized(nlp, corpus)
    print("Extraction terminée")

    # 3. Affichage des résultats
    for i in range(10):
        if i < len(results):
            print(results[i])