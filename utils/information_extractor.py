import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from datasets import load_dataset
from tqdm import tqdm
import re
import json
import argparse
from thinc.api import get_current_ops
from collections import defaultdict

class InformationExtractor:
    def __init__(self, nlp):
        self.nlp = nlp
        self.matcher = Matcher(nlp.vocab)
        self._add_patterns()

    def _add_patterns(self):
        """
        GOAL 3: Implement extraction patterns (Taxonomy, Jobs, Issues).
        We define patterns for spaCy's Matcher here.
        """
        
        # 1. TAXONOMY Pattern (Hyponymy)
        # English: "countries such as France", "players like Messi"
        self.matcher.add("IS_A", [
            [
                {"POS": "NOUN", "OP": "+"},      # Hypernym (e.g., "countries")
                {"LOWER": {"IN": ["such", "like", "including"]}}, # Trigger
                {"LOWER": "as", "OP": "?"},      # Optional 'as' for 'such as'
                {"ENT_TYPE": {"IN": ["GPE", "PERSON", "ORG"]}, "OP": "+"} # Hyponym
            ]
        ])

        # 2. JOB TITLE Pattern (FIXED)
        # English: "Foreign Minister Heiko Maas", "Heiko Maas, the Foreign Minister"
        self.matcher.add("JOB_TITLE", [
            [
                {"ENT_TYPE": "PERSON"},          # Person
                {"IS_PUNCT": True},              # comma
                {"POS": "DET", "OP": "?"},       # <--- FIXED: Changed from "DET" to "POS": "DET"
                {"POS": "NOUN", "OP": "+"},      # Title (e.g. minister)
                {"POS": "ADP", "OP": "?"},       # of
                {"POS": "NOUN", "OP": "*"}       # department (e.g. defense)
            ],
            [
                {"POS": "NOUN", "OP": "+"},      # Title
                {"ENT_TYPE": "PERSON"}           # Person
            ]
        ])

        # 3. SOCIAL CAUSE Pattern ("Lutte contre X")
        # English: "fight against pollution", "war on terror"
        self.matcher.add("SOCIAL_ISSUE", [
            [
                {"LOWER": {"IN": ["fight", "war", "battle", "struggle"]}},
                {"LOWER": {"IN": ["against", "on"]}},
                {"POS": "NOUN", "OP": "+"}       # The issue
            ]
        ])

    def normalize_entities(self, doc):
        """
        GOAL 1: Reduce noise.
        Simple heuristic: If a short entity (e.g., "Royant") is a substring 
        of a longer entity in the same doc (e.g., "Olivier Royant"), drop the short one.
        """
        entities = []
        seen_texts = set(e.text for e in doc.ents)
        
        for ent in doc.ents:
            # Check if this entity is a substring of another entity in the same doc
            is_noise = False
            for other_text in seen_texts:
                if ent.text != other_text and ent.text in other_text:
                    # It's likely a partial match (noise), skip it
                    is_noise = True
                    break
            
            if not is_noise:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        return entities

    def find_acronyms(self, doc):
        """
        GOAL 2: Acronyms and Long forms.
        Heuristic: Look for (ABC) and check if previous words match A... B... C...
        """
        acronyms = []
        # Regex to find parenthesized uppercase text, e.g., (AQLPA) or (GDP)
        # Matches: space, (, 2+ capital letters, )
        matches = re.finditer(r'\s\(([A-Z]{2,})\)', doc.text)
        
        for match in matches:
            short_form = match.group(1) # e.g., GDP
            span_end = match.start()
            
            # Look backwards in text to find the definition
            # This is a simplified implementation. 
            # A robust one looks at the first letter of preceding words.
            definition = self._scan_for_definition(doc.text, span_end, short_form)
            
            if definition:
                acronyms.append({
                    "short": short_form,
                    "long": definition
                })
        return acronyms

    def _scan_for_definition(self, text, end_index, acronym):
        """Backtracks from (Acronym) to find matching words."""
        words = text[:end_index].split()
        if len(words) < len(acronym):
            return None
            
        # Grab the last N words where N = len(acronym) (+ some buffer for 'of/the')
        window_size = len(acronym) + 2 
        candidates = words[-window_size:]
        
        # Very naive check: Do the first letters of the last few words match the acronym?
        # A real implementation requires complex alignment logic.
        # Here we just return the text window as a 'candidate definition'
        return " ".join(candidates)

    def extract_relations(self, doc):
        """Run the patterns defined in __init__"""
        relations = []
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            pattern_name = self.nlp.vocab.strings[match_id]
            relations.append({
                "type": pattern_name,
                "text": span.text
            })
        return relations