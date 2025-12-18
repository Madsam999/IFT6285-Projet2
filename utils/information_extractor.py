import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
import re

import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
import re

class InformationExtractor:
    def __init__(self, nlp):
        self.nlp = nlp
        self.matcher = Matcher(nlp.vocab)
        self._add_patterns()

    def _add_patterns(self):
        # 1. TAXONOMY (IS-A)
        self.matcher.add("TAXONOMY", [
            [
                {"POS": "NOUN", "OP": "+"},      
                {"LOWER": {"IN": ["such", "like", "including"]}},
                {"LOWER": "as", "OP": "?"},
                {"ENT_TYPE": {"IN": ["GPE", "ORG", "LOC"]}, "OP": "+"}
            ]
        ])

        # 2. SOCIAL ISSUES
        self.matcher.add("SOCIAL_ISSUE_CONFLICT", [
            [
                {"LOWER": {"IN": ["fight", "war", "battle", "struggle", "campaign", "movement"]}},
                {"LOWER": {"IN": ["against", "on", "for"]}},
                {"POS": "DET", "OP": "?"},
                {"POS": "ADJ", "OP": "*"}, 
                {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"} 
            ]
        ])

        self.matcher.add("SOCIAL_ISSUE_CRISIS", [
            [
                {"POS": "ADJ", "OP": "*"},
                {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}, 
                {"LOWER": {"IN": ["crisis", "epidemic", "shortage", "scandal", "inequality"]}}
            ]
        ])

        self.matcher.add("SOCIAL_ISSUE_ACTION", [
            [
                {"LEMMA": {"IN": ["tackle", "combat", "alleviate", "eradicate", "address"]}},
                {"POS": "DET", "OP": "?"},
                {"POS": "ADJ", "OP": "*"},
                {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}
            ]
        ])

    def _get_structured_context(self, span):
        token = span.root
        while token.pos_ not in ["VERB", "AUX"] and token.head != token:
            token = token.head
        for _ in range(5):
            if token.dep_ == "ccomp":
                token = token.head
                continue
            subjects = [t for t in token.children if t.dep_ in ["nsubj", "nsubjpass", "csubj"]]
            if subjects:
                subj_text = " ".join([t.text for t in subjects[0].subtree])
                return f"{subj_text} {token.text} ... {span.text}"
            if token.head == token: break
            token = token.head
        return f"Unknown source ... {span.text}"

    def _extract_job_titles_dependency(self, doc):
        titles = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Pre-nominal
                title_tokens = []
                valid_heads = set(range(ent.start, ent.end))
                idx = ent.start - 1
                while idx >= 0:
                    token = doc[idx]
                    if token.is_sent_start: break
                    if token.head.i in valid_heads and token.dep_ in ["compound", "amod", "nmod"]:
                        title_tokens.insert(0, token.text)
                        valid_heads.add(token.i)
                    else: break
                    idx -= 1
                if title_tokens and len(" ".join(title_tokens)) > 2:
                    titles.append({"type": "JOB_TITLE", "text": f"{' '.join(title_tokens)} {ent.text}"})

                # Post-nominal
                for child in ent.root.children:
                    if child.dep_ == "appos":
                        appos_span = doc[child.left_edge.i : child.right_edge.i + 1]
                        if any(t.pos_ in ["NOUN", "PROPN"] for t in appos_span):
                             titles.append({"type": "JOB_TITLE_APPOSITION", "text": f"{ent.text}, {appos_span.text}"})
        return titles

    def _extract_acronyms(self, doc):
        """
        Refined to catch:
        1. Backward: United Nations (UN)
        2. Forward: EPA (Environmental Agency)
        3. Standalone: "...see the FBI." (New)
        """
        acronyms = []
        # Keep track of token indices that are part of found acronyms to avoid duplicates
        seen_token_indices = set()

        # -------------------------------------------------------
        # CASE 1: Backward Search -> "United Nations (UN)"
        # -------------------------------------------------------
        pattern_backward = r'\s\(([A-Z][A-Za-z0-9.&-]*)\)'
        for match in re.finditer(pattern_backward, doc.text):
            short_form = match.group(1)
            span_start = match.start()
            
            # Mark indices as seen (approximate map from char to token)
            # We use a simple heuristic: if a span is found, we won't add standalone tokens from it.
            # (Strict token mapping is complex with regex, but we will filter duplicates at the end).
            
            clean_short = short_form.replace(".", "")
            preceding_text = doc.text[:span_start].strip()
            tokens = preceding_text.split()
            definition = "unknown"
            
            if len(tokens) >= len(clean_short):
                candidate_words = tokens[-len(clean_short):]
                initials = "".join([w[0].upper() for w in candidate_words])
                if initials == clean_short:
                    definition = " ".join(candidate_words)
                else:
                    candidate_words_ext = tokens[-min(len(tokens), len(clean_short) + 3):]
                    definition_candidate = " ".join(candidate_words_ext)
                    match_count = sum(1 for w in candidate_words_ext if w[0].upper() in clean_short)
                    if match_count >= len(clean_short) * 0.7:
                         definition = definition_candidate

            acronyms.append({"type": "ACRONYM", "acronym": short_form, "definition": definition})
            # Add the acronym string to a set to prevent adding it again as standalone
            seen_token_indices.add(short_form)

        # -------------------------------------------------------
        # CASE 2: Forward Search -> "EPA (Environmental...)"
        # -------------------------------------------------------
        pattern_forward = r'([A-Z0-9.&-]{2,})\s\(([^)]+)\)'
        for match in re.finditer(pattern_forward, doc.text):
            potential_acronym = match.group(1)
            
            # Avoid duplicate if Case 1 caught it
            if potential_acronym in seen_token_indices: continue

            potential_def = match.group(2)
            clean_acronym = potential_acronym.replace(".", "")
            
            def_words = potential_def.split()
            match_count = 0
            for char in clean_acronym:
                if char in [w[0].upper() for w in def_words]:
                    match_count += 1
            
            final_def = "unknown"
            if match_count >= len(clean_acronym) * 0.7:
                 final_def = potential_def
            elif len(clean_acronym) >= 3:
                 final_def = "unknown"
            else:
                 continue

            acronyms.append({"type": "ACRONYM", "acronym": potential_acronym, "definition": final_def})
            seen_token_indices.add(potential_acronym)

        # -------------------------------------------------------
        # CASE 3: Standalone -> "FBI" (No parens)
        # -------------------------------------------------------
        # We iterate through the spaCy tokens to find things that look like acronyms
        for token in doc:
            txt = token.text
            
            # Skip if we already processed this string in a regex match
            if txt in seen_token_indices:
                continue

            # CRITERIA:
            # 1. All Uppercase
            # 2. Length 2-6 (Avoids single letters "I" "A", and long noise)
            # 3. Is a Proper Noun or Noun (Avoids "IS", "AT", "US" if tagged as PRON)
            # 4. Check against a small blocklist of common confusing uppercase words
            blocklist = ["US", "IT", "OK", "AM", "PM", "TV", "PC"] 
            
            if token.is_upper and 2 <= len(txt) <= 6 and token.pos_ in ["PROPN", "NOUN"]:
                if txt not in blocklist:
                    acronyms.append({
                        "type": "ACRONYM",
                        "acronym": txt,
                        "definition": "unknown"
                    })
                    seen_token_indices.add(txt) # Prevent adding same token twice

        return acronyms

    def extract_relations(self, doc):
        relations = []
        matches = self.matcher(doc)
        spans = []
        for match_id, start, end in matches:
            span = doc[start:end]
            span.label_ = self.nlp.vocab.strings[match_id]
            spans.append(span)

        filtered = filter_spans(spans)
        for span in filtered:
            rel_type = "SOCIAL_ISSUE" if "SOCIAL_ISSUE" in span.label_ else span.label_
            context_str = None
            if rel_type == "SOCIAL_ISSUE":
                context_str = self._get_structured_context(span)
            result = {"type": rel_type, "text": span.text}
            if context_str: result["context"] = context_str
            relations.append(result)

        relations.extend(self._extract_job_titles_dependency(doc))
        return relations
    
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