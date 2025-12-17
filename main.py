import spacy
from datasets import load_dataset
from tqdm import tqdm
import re
import json
import argparse
from thinc.api import get_current_ops

# Optimized defaults for transformer NER (en_core_web_trf)
MODEL_NAME = "en_core_web_trf"
CHUNK_WORDS = 300
BATCH_SIZE = 256
N_PROCESS = 1
STORE_TEXT = True          # set to False to reduce file size / speed up (less disk I/O)
WRITE_BUFFER_SIZE = 512    # number of JSON lines buffered before writing


def iter_word_chunks(text: str, max_words: int):
    """Yield chunks of at most max_words words without building an intermediate list."""
    words = text.split()
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if chunk:
            yield chunk


def iter_corpus_chunks(dataset, max_articles=None, max_chunks=None, chunk_words=CHUNK_WORDS):
    """
    Itère sur le dataset et yield des chunks de texte, pour éviter de tout garder en RAM.
    """
    # dataset peut être un DatasetDict (avec split "train") ou un Dataset (split direct)
    texts = dataset["train"]["article"] if isinstance(dataset, dict) else dataset["article"]
    regex = re.compile(r"=\s=\s.+?\s=\s=")

    total_articles = len(texts)
    articles_to_process = min(max_articles, total_articles) if max_articles else total_articles

    if max_articles:
        print(f"Mode prototype: traitement de {articles_to_process:,} articles sur {total_articles:,} total")

    chunks_emitted = 0
    for i, text in enumerate(tqdm(texts, desc="Pré-traitement du texte", total=articles_to_process)):
        if max_articles and i >= max_articles:
            break
        text = text.strip()
        if not text:
            continue
        cleaned_text = regex.sub(" ", text).strip()
        if not cleaned_text:
            continue
        for chunk in iter_word_chunks(cleaned_text, max_words=chunk_words):
            yield chunk
            chunks_emitted += 1
            if max_chunks and chunks_emitted >= max_chunks:
                return


def extractNER_and_save_streaming(nlp, corpus_iter, path="ner_results.jsonl", batch_size=BATCH_SIZE):
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
        doc_generator = nlp.pipe(corpus_iter, batch_size=batch_size, n_process=N_PROCESS)

        # total inconnu en mode streaming -> tqdm sans total
        buffer = []
        for doc in tqdm(doc_generator, desc="Extraction des entités nommées"):
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                })

            record = {"entities": entities}
            if STORE_TEXT:
                record["text"] = doc.text

            buffer.append(json.dumps(record, ensure_ascii=False))
            if len(buffer) >= WRITE_BUFFER_SIZE:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()

        if buffer:
            f.write("\n".join(buffer) + "\n")

    print(f"Résultats sauvegardés dans: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extraction NER depuis WikiText103')
    parser.add_argument('--max-articles', type=int, default=None,
                       help='Nombre maximum d\'articles à traiter (pour prototypage rapide). Ex: 1000')
    parser.add_argument('--output', type=str, default='ner_results.jsonl',
                       help='Fichier de sortie (défaut: ner_results.jsonl)')
    
    args = parser.parse_args()
    
    # 1. Chargement des ressources
    print("Chargement du dataset WikiText103...")
    try:
        ds = load_dataset("Mouwiya/wikitext103-train", split="train")
    except TypeError:
        ds = load_dataset("Mouwiya/wikitext103-train")
    gpu_ok = spacy.prefer_gpu()
    print("spaCy GPU active:", gpu_ok)
    print("Backend Thinc:", get_current_ops())
    print(f"Chargement du modèle: {MODEL_NAME}")
    nlp = spacy.load(MODEL_NAME)
    
    # Désactiver les pipes non nécessaires
    if nlp.has_pipe("transformer"):
        pipes_to_keep = {"transformer", "ner"}
    elif nlp.has_pipe("tok2vec"):
        pipes_to_keep = {"tok2vec", "ner"}
    else:
        pipes_to_keep = {"ner"}
    pipes_to_disable = [pipe for pipe in nlp.pipe_names 
                       if pipe not in pipes_to_keep]
    if pipes_to_disable:
        nlp.disable_pipes(*pipes_to_disable)

    # Extraction NER et sauvegarde (streaming)
    print("\n" + "="*80)
    print("Extraction des NER du corpus")
    print("="*80)
    corpus_iter = iter_corpus_chunks(
        ds,
        max_articles=args.max_articles,
    )

    extractNER_and_save_streaming(
        nlp,
        corpus_iter,
        path=args.output,
        batch_size=BATCH_SIZE,
    )
    print(f"\nExtraction terminée. Résultats sauvegardés dans: {args.output}")
