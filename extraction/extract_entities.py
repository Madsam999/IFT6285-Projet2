"""
Étape 1: Extraction des entités nommées depuis le dataset.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy
from datasets import load_dataset
from tqdm import tqdm
import re
import json
import argparse
from thinc.api import get_current_ops

# Import des fonctions depuis main.py
from main import (
    MODEL_NAME, CHUNK_WORDS, BATCH_SIZE, N_PROCESS, 
    STORE_TEXT, WRITE_BUFFER_SIZE,
    iter_corpus_chunks, extractNER_and_save_streaming
)


def main():
    parser = argparse.ArgumentParser(description='Étape 1: Extraction des entités NER')
    parser.add_argument('--max-articles', type=int, default=None,
                       help='Nombre maximum d\'articles à traiter')
    parser.add_argument('--output', type=str, 
                       default='results/raw/entities_raw.jsonl',
                       help='Fichier de sortie')
    parser.add_argument('--dataset', type=str, default='Mouwiya/wikitext103-train',
                       help='Dataset à utiliser')
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Chargement des ressources
    print("="*80)
    print("ÉTAPE 1: EXTRACTION DES ENTITÉS NER")
    print("="*80)
    print(f"Chargement du dataset: {args.dataset}...")
    
    try:
        ds = load_dataset(args.dataset, split="train")
    except TypeError:
        ds = load_dataset(args.dataset)
    
    gpu_ok = spacy.prefer_gpu()
    print(f"spaCy GPU active: {gpu_ok}")
    print(f"Backend Thinc: {get_current_ops()}")
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
    
    # Extraction NER
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
    
    print(f"\n✓ Extraction terminée. Résultats sauvegardés dans: {args.output}")


if __name__ == '__main__':
    main()

