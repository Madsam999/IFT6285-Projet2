import pandas as pd
import re
from collections import Counter
import os
from evaluation.clean_entities import clean_entities
from evaluation.clean_relations import clean_relations

class Evaluator:
    def __init__(self):
        """
        Initializes the Evaluator. 
        You can add dictionaries or blocklists here if needed later.
        """
        self.results = {}
        self.cleaned_relations = clean_relations()
        self.cleaned_entities = clean_entities()
        self.stopwords = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'but', 'not'}

    def _check_cleanliness(self, text):
        """Internal helper: Returns True if text has artifacts like 's or trailing punctuation."""
        if not isinstance(text, str): return True
        # Check for 's at end (Tokenization artifact)
        if text.strip().endswith("'s"): return False
        # Check for leading punctuation (", Arkansas")
        if re.match(r'^[.,;:]', text.strip()): return False
        # Check for unbalanced parentheses
        if text.count('(') != text.count(')'): return False
        return True

    def evaluate_relations(self):
        """
        Evaluates the coherence of the relations dataframe.
        """
        print("--- EVALUATING RELATIONS ---")
        stats = {
            "total_rows": len(self.cleaned_relations),
            "cleanliness_issues": 0,
            "date_format_errors": 0,
            "ambiguous_subjects": 0
        }
        
        issues = []

        # 1. Cleanliness Check (Metric: Tokenization Artifacts)
        # We check both Subject and Object columns
        for index, row in self.cleaned_relations.iterrows():
            e1, e2 = row['entity1'], row['entity2']
            if not self._check_cleanliness(e1) or not self._check_cleanliness(e2):
                stats["cleanliness_issues"] += 1
                issues.append(f"Row {index}: Artifact detected in '{e1}' or '{e2}'")

        # 2. Logic Constraint: EVENT_DATE
        # Dates should look like years (4 digits) or full dates.
        date_rows = self.cleaned_relations[self.cleaned_relations['relation'] == 'EVENT_DATE']
        for index, row in date_rows.iterrows():
            date_str = str(row['entity2']).strip()
            # Simple regex for Year (18xx-20xx) or standard formats
            if not re.match(r'^(\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})$', date_str):
                stats["date_format_errors"] += 1
                issues.append(f"Row {index}: Invalid Date format '{date_str}'")

        # 3. Coherence Check: Subject Ambiguity
        # If we have "Totten" and "James Totten", the data is incoherent/duplicate.
        subjects = self.cleaned_relations['entity1'].unique()
        ambiguous_count = 0
        for subj in subjects:
            # Find if this subject is a substring of another subject (e.g. "Totten" in "James Totten")
            superstrings = [s for s in subjects if subj != s and subj in s]
            if superstrings:
                ambiguous_count += 1
                issues.append(f"Ambiguity: '{subj}' might be a duplicate of {superstrings}")
        
        stats["ambiguous_subjects"] = ambiguous_count

        # 4. Calculate Scores
        cleanliness_score = 100 - (stats["cleanliness_issues"] / stats["total_rows"] * 100)
        
        self.results['relations'] = stats
        
        # Report
        print(f"Total Relations: {stats['total_rows']}")
        print(f"Cleanliness Score: {cleanliness_score:.1f}%")
        print(f"Logic Errors (Dates): {stats['date_format_errors']}")
        print(f"Ambiguity Alerts: {stats['ambiguous_subjects']}")
        
        if issues:
            print("\n[Sample Issues Found]")
            for i in issues[:5]: # Show top 5
                print(f" - {i}")

    def _check_cleanliness(self, text):
        """Internal helper for relations."""
        if not isinstance(text, str): return True
        if text.strip().endswith("'s"): return False
        if re.match(r'^[.,;:]', text.strip()): return False
        if text.count('(') != text.count(')'): return False
        return True

    def _is_initials_match(self, acronym, definition):
        """
        Checks if the acronym letters appear in the definition words.
        Heuristic: At least 70% of acronym letters should match first letters of def words.
        """
        if not isinstance(definition, str) or not isinstance(acronym, str):
            return False
            
        def_words = [w for w in definition.split() if w[0].isalpha()]
        acro_chars = [c.lower() for c in acronym if c.isalpha()]
        
        matches = 0
        def_idx = 0
        
        for char in acro_chars:
            # Try to find this char as a start of a word in definition
            while def_idx < len(def_words):
                if def_words[def_idx].lower().startswith(char):
                    matches += 1
                    def_idx += 1 # Move to next word
                    break
                def_idx += 1
                
        # If >70% of acronym letters found as initials, it's coherent
        if len(acro_chars) == 0: return False
        return (matches / len(acro_chars)) >= 0.7
    
    def evaluate_entities(self):
        """
        Evaluates Entities for formatting, length, and ambiguity.
        Expects cols: entity;type;frequency (or similar)
        """
        print("\n--- EVALUATING ENTITIES ---")
        stats = {"total": len(self.cleaned_entities), "noise": 0, "outliers": 0, "ambiguous": 0}
        issues = []

        # 1. Check Noise & Length
        for idx, row in self.cleaned_entities.iterrows():
            text = str(row['entity'])
            
            # Noise Check
            if (len(text) < 2) or re.search(r'[\n\r;]', text) or re.match(r'^[^\w]', text):
                stats["noise"] += 1
                issues.append(f"Noise: '{text}' (Row {idx})")
            
            # Length Outlier Check (> 6 words is suspicious for NER)
            if len(text.split()) > 6:
                stats["outliers"] += 1
                issues.append(f"Length Outlier: '{text}' ({len(text.split())} words)")

        # 2. Check Ambiguity (Same text, different types)
        # Group by text and count unique types
        ambiguity_check = self.cleaned_entities.groupby('entity')['type'].nunique()
        ambiguous_entities = ambiguity_check[ambiguity_check > 1]
        stats["ambiguous"] = len(ambiguous_entities)
        
        if len(ambiguous_entities) > 0:
            issues.append(f"Ambiguous Types found: {ambiguous_entities.index.tolist()[:3]}...")

        print(f"Total Entities: {stats['total']}")
        print(f"Formatting Noise: {stats['noise']} ({(stats['noise']/stats['total'])*100:.1f}%)")
        print(f"Length Outliers: {stats['outliers']}")
        print(f"Type Ambiguity: {stats['ambiguous']} unique entities have multiple types")
        
        if issues:
            print("[Sample Entity Issues]")
            for i in issues[:3]: print(f" - {i}")

    def evaluate_acronyms(self, df: pd.DataFrame):
        """
        Evaluates Acronyms for validity and consistency.
        Expects cols: acronym;definition
        """
        print("\n--- EVALUATING ACRONYMS ---")
        stats = {"total": len(df), "stopword_acronyms": 0, "mismatch_logic": 0}
        issues = []

        for idx, row in df.iterrows():
            acr = str(row['acronym'])
            defn = str(row['full_form'])

            # 1. Stopword Check (Is 'THE' tagged as acronym?)
            if acr.lower() in self.stopwords:
                stats["stopword_acronyms"] += 1
                issues.append(f"Stopword Alert: '{acr}' is likely not an acronym")

            # 2. Logic Check (Does EPA match Environmental...?)
            # Skip this check if definition is "Unknown" or "Contextual"
            if "unknown" not in defn.lower() and "contextual" not in defn.lower():
                if not self._is_initials_match(acr, defn):
                    stats["mismatch_logic"] += 1
                    issues.append(f"Logic Mismatch: '{acr}' != '{defn}'")
                    

# ==========================================
# TEST RUN
# ==========================================
if __name__ == "__main__":
    from io import StringIO
    
    csv_data = os

    df_test = pd.read_csv(StringIO(csv_data), sep=';')
    
    evaluator = Evaluator()
    evaluator.evaluate_relations(df_test)