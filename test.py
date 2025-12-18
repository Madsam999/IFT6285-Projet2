import spacy
from datasets import load_dataset
from tqdm import tqdm
import re
import json
import argparse
from thinc.api import get_current_ops
from utils.information_extractor import InformationExtractor

# Optimized defaults for transformer NER (en_core_web_trf)
MODEL_NAME = "en_core_web_trf"
CHUNK_WORDS = 300
BATCH_SIZE = 256
N_PROCESS = 1
STORE_TEXT = True          # set to False to reduce file size / speed up (less disk I/O)
WRITE_BUFFER_SIZE = 512    # number of JSON lines buffered before writing

# --- EXISTING HELPER FUNCTIONS (UNCHANGED) ---
def iter_word_chunks(text: str, max_words: int):
    words = text.split()
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if chunk: yield chunk

def iter_corpus_chunks(dataset, max_articles=None, chunk_words=CHUNK_WORDS):
    texts = dataset["train"]["article"] if isinstance(dataset, dict) else dataset["article"]
    regex = re.compile(r"=\s=\s.+?\s=\s=")
    total = min(max_articles, len(texts)) if max_articles else len(texts)
    
    for i, text in enumerate(tqdm(texts, total=total, desc="Preprocessing")):
        if max_articles and i >= max_articles: break
        text = regex.sub(" ", text).strip()
        if text:
            for chunk in iter_word_chunks(text, max_words=chunk_words):
                yield chunk

# --- UPDATED SAVING FUNCTION ---
def extract_and_save(nlp, corpus_iter, path="extracted_data.jsonl"):
    print("Initializing Information Extraction Pipeline...")
    extractor = InformationExtractor(nlp)
    
    with open(path, "w", encoding="utf-8") as f:
        # Use nlp.pipe for efficiency
        doc_generator = nlp.pipe(corpus_iter, batch_size=BATCH_SIZE)
        
        buffer = []
        for doc in tqdm(doc_generator, desc="Extracting Information"):
            
            # 1. Clean Entities
            clean_ents = extractor.normalize_entities(doc)
            
            # 2. Acronyms
            acronyms = extractor._extract_acronyms(doc)
            
            # 3. Relations (Patterns)
            relations = extractor.extract_relations(doc)
            
            # Only save if we found something interesting
            if clean_ents or acronyms or relations:
                record = {
                    "text_snippet": doc.text[:50] + "...", # Preview only to save space
                    "entities": clean_ents,
                    "acronyms": acronyms,
                    "relations": relations
                }
                buffer.append(json.dumps(record, ensure_ascii=False))

            if len(buffer) >= WRITE_BUFFER_SIZE:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()
        
        if buffer:
            f.write("\n".join(buffer) + "\n")

# --- MAIN BLOCK ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-articles', type=int, default=100)
    args = parser.parse_args()

    print(f"Loading Model: {MODEL_NAME}")
    # Load model (disable parser/lemmatizer if you ONLY wanted NER, 
    # but for Patterns using POS tags, we NEED the parser or tagger)
    nlp = spacy.load(MODEL_NAME) 

    # Load Data
    print("Loading WikiText103...")
    try:
        ds = load_dataset("Mouwiya/wikitext103-train", split="train")
    except:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")

    # Run Pipeline
    corpus = iter_corpus_chunks(ds, max_articles=50)
    extract_and_save(nlp, corpus, path="results_ie.jsonl")
    print("Done! Results saved to results_ie.jsonl")