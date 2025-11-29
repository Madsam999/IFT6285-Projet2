import spacy
from datasets import load_dataset
from tqdm import tqdm
import re
import json
from thinc.api import get_current_ops


def extract_named_entities(model, corpus):
    for sentence in tqdm(corpus, desc="Extraction des entitées nommées"):
        doc = model(sentence)


def split_long_text(text, max_len=300):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+max_len])
        chunks.append(chunk)
        i += max_len
    return chunks


def preprocess_corpus(dataset):
    texts = dataset["train"]["article"]
    regexPattern = r"=\s=\s.+?\s=\s="

    sentences = []
    for text in tqdm(texts, desc="Pré-traitement du texte"):
        if not text.strip():
            continue
        cleaned_text = re.sub(regexPattern, " ", text)
        preprocessedText = cleaned_text.strip().lower()
        if preprocessedText:
            chunks = split_long_text(preprocessedText)
            sentences.extend(chunks)
    return sentences


def extractNER_and_save_streaming(nlp, corpus, path="ner_results.jsonl"):
    """
    Pour chaque chunk du corpus, on sauvegarde une ligne JSON de la forme :
    {
      "text": "...",
      "entities": [
        {"text": "...", "label": "...", "start_char": 0, "end_char": 4},
        ...
      ]
    }
    """
    print("Début de l'extraction + sauvegarde streaming...")

    with open(path, "w", encoding="utf-8") as f:
        doc_generator = nlp.pipe(
            corpus,
            batch_size=64,
            n_process=1
        )

        for doc in tqdm(doc_generator, desc="Extraction des entités nommées", total=len(corpus)):
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                })

            record = {
                "text": doc.text,
                "entities": entities
            }

            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"Résultats sauvegardés dans: {path}")


if __name__ == '__main__':
    # 1. Chargement des ressources
    ds = load_dataset("Mouwiya/wikitext103-train")
    gpu_ok = spacy.prefer_gpu()
    print("spaCy GPU active:", gpu_ok)
    print("Backend Thinc:", get_current_ops())
    nlp = spacy.load("en_core_web_trf")

    # désactiver tout sauf le NER
    for pipe_name in nlp.pipe_names:
        if pipe_name != "ner":
            nlp.disable_pipes(pipe_name)
    print("Pipes actifs :", nlp.pipe_names)

    # 2. Pré-traitement
    print("Début du pré-traitement du dataset Wikitext103-train")
    corpus = preprocess_corpus(ds)
    print(f"Pré-traitement terminé. {len(corpus)} documents à traiter.")

    # 3. Extraction NER et sauvegarde (streaming)
    print("Extraction des NER du corpus (Optimisée avec nlp.pipe)")
    extractNER_and_save_streaming(nlp, corpus, path="ner_results.jsonl")
    print("Extraction terminée")
