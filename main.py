import spacy
from datasets import load_dataset
from tqdm import tqdm
import re
import json
import argparse
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


def preprocess_corpus(dataset, lowercase=False, max_articles=None):
    """
    Pré-traite le corpus.
    
    Args:
        dataset: Dataset HuggingFace
        lowercase: Si True, convertit en minuscules (déconseillé pour NER)
        max_articles: Nombre maximum d'articles à traiter (None = tous)
    """
    texts = dataset["train"]["article"]
    regexPattern = r"=\s=\s.+?\s=\s="

    sentences = []
    total_articles = len(texts)
    articles_to_process = min(max_articles, total_articles) if max_articles else total_articles
    
    if max_articles:
        print(f"Mode prototype: traitement de {articles_to_process:,} articles sur {total_articles:,} total")
    
    for i, text in enumerate(tqdm(texts, desc="Pré-traitement du texte", total=articles_to_process)):
        if max_articles and i >= max_articles:
            break
        if not text.strip():
            continue
        cleaned_text = re.sub(regexPattern, " ", text)
        preprocessedText = cleaned_text.strip()
        if lowercase:
            preprocessedText = preprocessedText.lower()
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
    parser = argparse.ArgumentParser(description='Extraction NER depuis WikiText103')
    parser.add_argument('--max-articles', type=int, default=None,
                       help='Nombre maximum d\'articles à traiter (pour prototypage rapide). Ex: 1000')
    parser.add_argument('--output', type=str, default='ner_results.jsonl',
                       help='Fichier de sortie (défaut: ner_results.jsonl)')
    parser.add_argument('--lowercase', action='store_true',
                       help='Convertir en minuscules (déconseillé pour NER)')
    parser.add_argument('--model', type=str, default='en_core_web_trf',
                       help='Modèle spaCy à utiliser (défaut: en_core_web_trf)')
    
    args = parser.parse_args()
    
    # 1. Chargement des ressources
    print("Chargement du dataset WikiText103...")
    ds = load_dataset("Mouwiya/wikitext103-train")
    gpu_ok = spacy.prefer_gpu()
    print("spaCy GPU active:", gpu_ok)
    print("Backend Thinc:", get_current_ops())
    print(f"Chargement du modèle: {args.model}")
    nlp = spacy.load(args.model)

    # Configuration des pipes selon le type de modèle
    if "trf" in args.model:
        # Modèle transformer-based: besoin de transformer, tagger, parser
        pipes_to_keep = {"transformer", "tagger", "parser", "ner"}
        print("Modele transformer detecte: garde transformer, tagger, parser, ner")
    else:
        # Modèles statistiques: seulement NER nécessaire
        pipes_to_keep = {"ner"}
        print("Modele statistique detecte: garde seulement NER")
    
    # Désactiver les pipes non nécessaires
    pipes_to_disable = [pipe for pipe in nlp.pipe_names 
                       if pipe not in pipes_to_keep]
    if pipes_to_disable:
        nlp.disable_pipes(*pipes_to_disable)
    
    print("Pipes actifs :", nlp.pipe_names)

    # 2. Pré-traitement
    print("\n" + "="*80)
    print("Début du pré-traitement du dataset Wikitext103-train")
    if not args.lowercase:
        print("ATTENTION: Le texte n'est PAS converti en minuscules pour préserver")
        print("la capitalisation nécessaire à la détection NER.")
    else:
        print("ATTENTION: Le texte sera converti en minuscules (peut affecter la qualité NER).")
    print("="*80)
    
    corpus = preprocess_corpus(ds, lowercase=args.lowercase, max_articles=args.max_articles)
    print(f"\nPré-traitement terminé. {len(corpus):,} chunks à traiter.")

    # 3. Extraction NER et sauvegarde (streaming)
    print("\n" + "="*80)
    print("Extraction des NER du corpus (Optimisée avec nlp.pipe)")
    print("="*80)
    extractNER_and_save_streaming(nlp, corpus, path=args.output)
    print(f"\nExtraction terminée. Résultats sauvegardés dans: {args.output}")
