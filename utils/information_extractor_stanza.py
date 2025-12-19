"""
Extracteur d'information utilisant Stanza au lieu de spaCy.
Stanza offre un meilleur parsing de dépendances, ce qui peut améliorer l'extraction de relations.
"""
import stanza
from stanza.models.common.doc import Document
import re
from typing import List, Dict
import wikipedia


class InformationExtractorStanza:
    def __init__(self, lang='en', processors='tokenize,pos,lemma,ner,depparse'):
        """
        Initialise Stanza avec les processeurs nécessaires.
        Note: lemma est nécessaire pour depparse, donc on doit le garder.
        On utilise word.text quand possible pour éviter les appels à lemma.
        
        Args:
            lang: Langue (défaut: 'en')
            processors: Processeurs à utiliser (défaut: tokenize,pos,lemma,ner,depparse)
        """
        print(f"Initialisation de Stanza (lang={lang}, processors={processors})...")
        self.nlp = stanza.Pipeline(lang=lang, processors=processors, use_gpu=True)
        print("Stanza initialisé avec succès.")
        self.blocklist = {"ID", "OK", "AM", "PM", "TV", "PC", "CD", "DJ", "MC", "US", "UK", "II", "III", "IV", "VI"}
        self.acronym_cache = {}
    
    def _extract_occupation_from_dependencies(self, doc: Document) -> List[Dict]:
        """
        Extrait les relations OCCUPATION en utilisant les dépendances Stanza.
        Stanza a un meilleur parsing de dépendances que spaCy.
        """
        relations = []
        
        # Créer un mapping des entités par position dans le texte
        entity_map = {}
        for ent in doc.entities:
            if ent.type == 'PERSON':
                # Stocker la position de début pour chaque entité
                for i in range(ent.start_char, ent.end_char):
                    entity_map[i] = ent
        
        for sentence in doc.sentences:
            # Chercher les patterns: "Title Person" ou "Person, Title"
            for word in sentence.words:
                # Pattern 1: "Title Person" - chercher un NOUN/PROPN avant une entité PERSON
                if word.upos in ['NOUN', 'PROPN']:
                    # Vérifier si c'est un titre (ex: "General", "President")
                    title_words = ['general', 'president', 'governor', 'mayor', 'director', 
                                  'manager', 'chief', 'captain', 'doctor', 'professor']
                    # Utiliser lemma pour une meilleure correspondance (gère les pluriels, etc.)
                    if word.lemma.lower() in title_words or word.text.lower() in title_words:
                        # Chercher une entité PERSON après ce mot
                        for ent in doc.entities:
                            if ent.type == 'PERSON' and ent.start_char > word.end_char:
                                # Vérifier qu'ils sont proches (dans la même phrase)
                                if ent.start_char - word.end_char < 50:  # Distance raisonnable
                                    relations.append({
                                        'type': 'OCCUPATION',
                                        'text': f"{word.text} {ent.text}",
                                        'person': ent.text,
                                        'title': word.text
                                    })
                                    break
                
                # Pattern 2: "Person, Title" - apposition (fusionné avec OCCUPATION)
                if word.deprel == 'appos':
                    # Chercher si le head est une entité PERSON
                    head_idx = word.head - 1  # head est 1-indexed
                    if head_idx >= 0 and head_idx < len(sentence.words):
                        head_word = sentence.words[head_idx]
                        # Vérifier si le head est dans une entité PERSON
                        for ent in doc.entities:
                            if ent.type == 'PERSON' and ent.start_char <= head_word.start_char < ent.end_char:
                                if word.upos in ['NOUN', 'PROPN']:
                                    relations.append({
                                        'type': 'OCCUPATION',  # Fusionné: même type que pré-nominal
                                        'text': f"{ent.text}, {word.text}",
                                        'person': ent.text,
                                        'title': word.text
                                    })
                                    break
        
        return relations
    
    
    def _extract_location_relations(self, doc: Document) -> List[Dict]:
        """Extrait les relations de localisation en utilisant les dépendances."""
        relations = []
        
        # Créer des maps d'entités par type
        person_entities = [e for e in doc.entities if e.type == 'PERSON']
        org_entities = [e for e in doc.entities if e.type == 'ORG']
        location_entities = [e for e in doc.entities if e.type in ['GPE', 'LOC', 'FAC']]
        
        for sentence in doc.sentences:
            for word in sentence.words:
                # Pattern: "Person born in Location"
                # Utiliser lemma pour gérer les formes conjuguées (born, bear, etc.)
                if word.lemma.lower() in ['born', 'birth', 'bear'] and word.upos == 'VERB':
                    # Chercher le sujet (personne) et l'objet (lieu) dans les entités
                    person = None
                    location = None
                    
                    # Chercher une entité PERSON avant le verbe
                    for ent in person_entities:
                        if ent.end_char < word.start_char and word.start_char - ent.end_char < 30:
                            person = ent.text
                            break
                    
                    # Chercher une entité LOCATION après le verbe
                    for ent in location_entities:
                        if ent.start_char > word.end_char and ent.start_char - word.end_char < 50:
                            location = ent.text
                            break
                    
                    if person and location:
                        relations.append({
                            'type': 'LOCATION_BORN',
                            'text': f"{person} born in {location}",
                            'entity1': person,
                            'entity2': location
                        })
                
                # Pattern: "Organization based in Location"
                # Utiliser lemma pour gérer les formes conjuguées
                if word.lemma.lower() in ['base', 'locate', 'headquarter'] or word.text.lower() in ['based', 'located', 'headquartered']:
                    org = None
                    location = None
                    
                    # Chercher une entité ORG ou PERSON avant
                    for ent in org_entities + person_entities:
                        if ent.end_char < word.start_char and word.start_char - ent.end_char < 30:
                            org = ent.text
                            break
                    
                    # Chercher une entité LOCATION après
                    for ent in location_entities:
                        if ent.start_char > word.end_char and ent.start_char - word.end_char < 50:
                            location = ent.text
                            break
                    
                    if org and location:
                        relations.append({
                            'type': 'LOCATION_BASED',
                            'text': f"{org} based in {location}",
                            'entity1': org,
                            'entity2': location
                        })
        
        return relations
    
    def _extract_org_member_relations(self, doc: Document) -> List[Dict]:
        """Extrait les relations organisationnelles."""
        relations = []
        
        person_entities = [e for e in doc.entities if e.type == 'PERSON']
        org_entities = [e for e in doc.entities if e.type == 'ORG']
        
        for sentence in doc.sentences:
            for word in sentence.words:
                # Utiliser lemma pour gérer les pluriels et formes conjuguées
                if word.lemma.lower() in ['member', 'founder', 'president', 'director', 'chairman', 'ceo'] or word.text.lower() in ['member', 'founder', 'president', 'director', 'chairman', 'ceo']:
                    person = None
                    org = None
                    
                    # Chercher une entité PERSON avant
                    for ent in person_entities:
                        if ent.end_char < word.start_char and word.start_char - ent.end_char < 30:
                            person = ent.text
                            break
                    
                    # Chercher une entité ORG après (dans "X of Y" ou "X at Y")
                    for ent in org_entities:
                        if ent.start_char > word.end_char and ent.start_char - word.end_char < 50:
                            org = ent.text
                            break
                    
                    if person and org:
                        relations.append({
                            'type': 'ORG_MEMBER',
                            'text': f"{person} {word.text} of {org}",
                            'entity1': person,
                            'entity2': org
                        })
        
        return relations
    
    def _resolve_acronym_via_wikipedia(self, acronym: str) -> str:
        """
        Helper: Tente de trouver la définition via Wikipedia si elle n'est pas dans le texte.
        Utilise un cache pour éviter de bannir l'IP.
        """
        # 1. Vérifier le cache
        if acronym in self.acronym_cache:
            return self.acronym_cache[acronym]

        # 2. Vérifier si c'est un mot commun ignoré
        if acronym in self.blocklist:
            return "Unknown"

        try:
            # Récupérer le titre de la page (Wikipedia gère les redirections ex: WHO -> World Health Organization)
            # auto_suggest=False est important pour éviter les corrections bizarres
            page_title = wikipedia.page(acronym, auto_suggest=False).title
            
            # Nettoyage: "EPA (Agency)" -> "EPA"
            clean_def = page_title.split('(')[0].strip()
            
            # Stocker dans le cache
            self.acronym_cache[acronym] = clean_def
            return clean_def

        except wikipedia.exceptions.DisambiguationError as e:
            # En cas d'ambiguïté, on prend la première option (la plus commune)
            first_option = e.options[0]
            self.acronym_cache[acronym] = first_option
            return first_option
            
        except (wikipedia.exceptions.PageError, Exception):
            # En cas d'erreur ou page non trouvée
            self.acronym_cache[acronym] = "Unknown"
            return "Unknown"

    def _extract_acronyms_with_stanza(self, doc: Document, text: str) -> List[Dict]:
        """
        Extrait les acronymes en utilisant une approche hybride :
        1. Contextuelle : Cherche les définitions explicites dans le texte (ex: "EPA (Environmental Protection Agency)").
        2. Standalone : Pour les acronymes sans définition, utilise Stanza (POS tags) + Wikipedia.
        """
        acronyms = []
        seen_acronyms = set()
        
        # -------------------------------------------------------
        # PHASE 1 : Recherche Contextuelle (Haute Confiance)
        # -------------------------------------------------------
        # Pattern: "Environmental Protection Agency (EPA)"
        # On cherche : Mot (MAJUSCULE)
        pattern_backward = r'\s\(([A-Z0-9][A-Za-z0-9.&-]{1,})\)'
        
        for match in re.finditer(pattern_backward, text):
            short_form = match.group(1)
            span_start = match.start()
            
            clean_short = short_form.replace(".", "").replace("-", "")
            if len(clean_short) < 2 or clean_short in self.blocklist:
                continue

            # 1. Look at text immediately before the acronym
            preceding_text = text[:span_start].strip()
            tokens = preceding_text.split()
            
            definition = "Unknown"
            
            # 2. Logic: Grab the last N words, where N is the length of the acronym
            # (e.g. for EPA (3 letters), grab last 3 words: "Environmental Protection Agency")
            if len(tokens) >= len(clean_short):
                # We grab a few extra words just in case (e.g. "The World Health...")
                # clean_short length + 2 is a safe window
                window = min(len(tokens), len(clean_short) + 2)
                candidate_words = tokens[-window:]
                
                # Simple Heuristic: match the first letters
                # Check the last 3 words, then last 4, etc.
                found_match = False
                for i in range(len(clean_short), window + 1):
                    subset = candidate_words[-i:]
                    # Get initials of this subset
                    initials = "".join([w[0].upper() for w in subset if w[0].isalpha()])
                    
                    # Fuzzy match: If initials contain the acronym (e.g. WHOrgc matches WHO)
                    if clean_short in initials:
                        definition = " ".join(subset)
                        found_match = True
                        break
                
                if not found_match:
                    # Fallback: Just take the last N words if no perfect match found
                    definition = " ".join(tokens[-len(clean_short):])

            acronyms.append({
                "type": "ACRONYM", 
                "acronym": short_form, 
                "definition": definition  # <--- Now holds the actual words!
            })
            seen_acronyms.add(short_form)

        # -------------------------------------------------------
        # PHASE 2 : Recherche Standalone via Stanza + Wikipedia
        # -------------------------------------------------------
        for sentence in doc.sentences:
            for word in sentence.words:
                txt = word.text
                clean_txt = txt.replace(".", "").replace("-", "")

                # Skip si déjà trouvé en Phase 1
                if txt in seen_acronyms:
                    continue

                # HEURISTIQUES STANZA :
                # 1. Tout en majuscules
                # 2. Longueur 2 à 6
                # 3. Tag POS est PROPN (Nom Propre) - Élimine "US" (pronoun), "IT" (pronoun) souvent
                # 4. Pas dans la blocklist
                if (txt.isupper() and 
                    2 <= len(clean_txt) <= 6 and 
                    word.upos == 'PROPN' and 
                    clean_txt not in self.blocklist):
                    
                    # C'est un candidat solide (ex: "FBI", "NASA")
                    # On demande à Wikipedia de le résoudre
                    definition = self._resolve_acronym_via_wikipedia(txt)
                    
                    acronyms.append({
                        "type": "ACRONYM",
                        "acronym": txt,
                        "definition": definition
                    })
                    seen_acronyms.add(txt)

        return acronyms
    
    def _extract_taxonomy_relations(self, doc: Document) -> List[Dict]:
        """Extrait les relations TAXONOMY (IS-A)."""
        relations = []
        
        # Pattern: "X such as Y", "X like Y", "X including Y"
        keywords = ['such', 'like', 'including']
        target_entity_types = ['GPE', 'ORG', 'LOC']
        
        for sentence in doc.sentences:
            for word in sentence.words:
                # Utiliser lemma pour une meilleure correspondance
                if word.lemma.lower() in keywords or word.text.lower() in keywords:
                    # Chercher une entité après le mot-clé
                    for ent in doc.entities:
                        if ent.type in target_entity_types:
                            # Vérifier que l'entité est après le mot-clé et proche
                            if ent.start_char > word.end_char and ent.start_char - word.end_char < 100:
                                # Chercher le concept général avant (dans les 50 caractères)
                                concept = 'UNKNOWN'
                                # Chercher un NOUN avant le mot-clé
                                for w in sentence.words:
                                    if w.end_char < word.start_char and word.start_char - w.end_char < 50:
                                        if w.upos == 'NOUN':
                                            concept = w.text
                                            break
                                
                                relations.append({
                                    'type': 'TAXONOMY',
                                    'text': f"{concept} {word.text} {ent.text}",
                                    'entity1': concept,
                                    'entity2': ent.text
                                })
                                break
        
        return relations
    
    def _extract_social_issue_relations(self, doc: Document) -> List[Dict]:
        """Extrait les relations SOCIAL_ISSUE (conflits, crises, actions)."""
        relations = []
        
        # SOCIAL_ISSUE_CONFLICT: "fight/war/battle against X"
        conflict_verbs = ['fight', 'war', 'battle', 'struggle', 'campaign', 'movement']
        conflict_preps = ['against', 'on', 'for']
        
        # SOCIAL_ISSUE_CRISIS: "X crisis/epidemic/shortage"
        crisis_nouns = ['crisis', 'epidemic', 'shortage', 'scandal', 'inequality']
        
        # SOCIAL_ISSUE_ACTION: "tackle/combat/alleviate X"
        action_verbs = ['tackle', 'combat', 'alleviate', 'eradicate', 'address']
        
        for sentence in doc.sentences:
            for word in sentence.words:
                # Pattern 1: CONFLICT - "fight against X"
                # Utiliser lemma pour gérer les formes conjuguées
                if word.lemma.lower() in conflict_verbs or word.text.lower() in conflict_verbs:
                    # Chercher une préposition "against/on/for" dans les enfants
                    for w in sentence.words:
                        if w.head == word.id and w.text.lower() in conflict_preps:
                            # Chercher un NOUN/PROPN après la préposition (dans les enfants de la préposition)
                            for w2 in sentence.words:
                                if w2.head == w.id and w2.upos in ['NOUN', 'PROPN']:
                                    # Construire le texte complet
                                    issue_text = w2.text
                                    # Chercher les mots composés
                                    for w3 in sentence.words:
                                        if w3.head == w2.id and w3.deprel in ['amod', 'compound']:
                                            issue_text = f"{w3.text} {issue_text}"
                                    
                                    relations.append({
                                        'type': 'SOCIAL_ISSUE_CONFLICT',
                                        'text': f"{word.text} {w.text} {issue_text}",
                                        'entity1': 'UNKNOWN',
                                        'entity2': issue_text
                                    })
                                    break
                
                # Pattern 2: CRISIS - "X crisis"
                # Utiliser lemma pour gérer les pluriels
                if word.lemma.lower() in crisis_nouns or word.text.lower() in crisis_nouns:
                    # Chercher un NOUN/PROPN avant (modificateur)
                    for w in sentence.words:
                        if w.head == word.id and w.upos in ['NOUN', 'PROPN', 'ADJ']:
                            issue_text = f"{w.text} {word.text}"
                            relations.append({
                                'type': 'SOCIAL_ISSUE_CRISIS',
                                'text': issue_text,
                                'entity1': 'UNKNOWN',
                                'entity2': issue_text
                            })
                            break
                
                # Pattern 3: ACTION - "tackle X"
                # Utiliser lemma pour gérer les formes conjuguées
                if word.lemma.lower() in action_verbs or word.text.lower() in action_verbs:
                    # Chercher un NOUN/PROPN après (objet direct ou indirect)
                    for w in sentence.words:
                        if w.head == word.id and w.deprel in ['obj', 'obl'] and w.upos in ['NOUN', 'PROPN']:
                            issue_text = w.text
                            # Chercher les modificateurs
                            for w2 in sentence.words:
                                if w2.head == w.id and w2.deprel in ['amod', 'compound']:
                                    issue_text = f"{w2.text} {issue_text}"
                            
                            relations.append({
                                'type': 'SOCIAL_ISSUE_ACTION',
                                'text': f"{word.text} {issue_text}",
                                'entity1': 'UNKNOWN',
                                'entity2': issue_text
                            })
                            break
        
        return relations
    
    def _extract_event_date_relations(self, doc: Document) -> List[Dict]:
        """Extrait les relations EVENT_DATE."""
        relations = []
        
        event_entities = [e for e in doc.entities if e.type == 'EVENT']
        date_entities = [e for e in doc.entities if e.type == 'DATE']
        
        # Pattern 1: "EVENT in DATE"
        for event in event_entities:
            for date in date_entities:
                # Vérifier qu'ils sont proches (dans les 100 caractères)
                distance = abs(event.start_char - date.start_char)
                if distance < 100:
                    relations.append({
                        'type': 'EVENT_DATE',
                        'text': f"{event.text} {date.text}",
                        'entity1': event.text,
                        'entity2': date.text
                    })
        
        # Pattern 2: "DATE EVENT" (inverse)
        for date in date_entities:
            for event in event_entities:
                # DATE avant EVENT
                if event.start_char > date.end_char and event.start_char - date.end_char < 100:
                    relations.append({
                        'type': 'EVENT_DATE',
                        'text': f"{date.text} {event.text}",
                        'entity1': event.text,
                        'entity2': date.text
                    })
        
        return relations
    
    def _extract_geographic_relations(self, doc: Document) -> List[Dict]:
        """Extrait les relations géographiques entre lieux."""
        relations = []
        
        location_entities = [e for e in doc.entities if e.type in ['GPE', 'LOC']]
        location_keywords = ['in', 'near', 'south', 'north', 'east', 'west', 'of']
        
        for sentence in doc.sentences:
            for word in sentence.words:
                # Les mots de localisation sont généralement invariables, mais utiliser lemma pour cohérence
                if word.lemma.lower() in location_keywords or word.text.lower() in location_keywords:
                    # Chercher deux entités LOCATION autour du mot-clé
                    loc1 = None
                    loc2 = None
                    
                    # Entité avant
                    for ent in location_entities:
                        if ent.end_char < word.start_char and word.start_char - ent.end_char < 50:
                            loc1 = ent
                            break
                    
                    # Entité après
                    for ent in location_entities:
                        if ent.start_char > word.end_char and ent.start_char - word.end_char < 50:
                            loc2 = ent
                            break
                    
                    if loc1 and loc2 and loc1.text != loc2.text:
                        relations.append({
                            'type': 'GEOGRAPHIC_RELATION',
                            'text': f"{loc1.text} {word.text} {loc2.text}",
                            'entity1': loc1.text,
                            'entity2': loc2.text
                        })
        
        return relations
    
    def extract_relations(self, text: str) -> List[Dict]:
        """
        Extrait toutes les relations depuis un texte.
        Couvre toutes les relations extraites par spaCy.
        
        Args:
            text: Texte à analyser
        
        Returns:
            Liste de dictionnaires avec les relations extraites
        """
        # Traiter le texte avec Stanza
        doc = self.nlp(text)
        
        relations = []
        
        # Extraire tous les types de relations (même que spaCy)
        relations.extend(self._extract_occupation_from_dependencies(doc))
        relations.extend(self._extract_location_relations(doc))
        relations.extend(self._extract_org_member_relations(doc))
        relations.extend(self._extract_taxonomy_relations(doc))
        relations.extend(self._extract_social_issue_relations(doc))
        relations.extend(self._extract_event_date_relations(doc))
        relations.extend(self._extract_geographic_relations(doc))
        
        return relations
    
    def extract_acronyms(self, text: str) -> List[Dict]:
        """Extrait les acronymes depuis un texte en utilisant Stanza."""
        doc = self.nlp(text)
        return self._extract_acronyms_with_stanza(doc, text)
    
    def normalize_entities(self, text: str) -> List[Dict]:
        """
        Normalise les entités extraites par Stanza.
        Dans Stanza, les entités sont dans doc.entities, pas dans word.ner.
        """
        doc = self.nlp(text)
        entities = []
        seen_texts = set()
        
        # Dans Stanza, les entités sont dans doc.entities
        for ent in doc.entities:
            entity_text = ent.text
            
            # Vérifier si c'est un sous-ensemble d'une autre entité
            is_noise = False
            for other_text in seen_texts:
                if entity_text != other_text and entity_text in other_text:
                    is_noise = True
                    break
            
            if not is_noise:
                entities.append({
                    "text": entity_text,
                    "label": ent.type,  # Type d'entité (PERSON, ORG, GPE, etc.)
                    "start": ent.start_char if hasattr(ent, 'start_char') else 0,
                    "end": ent.end_char if hasattr(ent, 'end_char') else len(entity_text)
                })
                seen_texts.add(entity_text)
        
        return entities

