"""
Export des résultats vers CSV (format requis par le projet).
Génère les fichiers CSV pour les entités, relations, et acronymes.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm
from utils.data_loader import load_ner_data


def export_entities_to_csv(input_file: str, output_file: str):
    """
    Exporte les entités vers un fichier CSV.
    Format: entity;type;frequency
    Filtre les entités avec bruit de formatage (trop courtes, caractères spéciaux).
    """
    import re
    print("Collecte des entités...")
    entity_counter = Counter()
    entity_types = {}
    filtered_count = 0
    
    for doc in tqdm(load_ner_data(input_file), desc="Lecture des documents"):
        for entity in doc.get('entities', []):
            text = entity.get('text', '')
            label = entity.get('label', 'UNKNOWN')
            
            if text:
                # Filtrer les entités avec bruit de formatage
                # Rejeter si trop court (< 2 caractères), contient retours à la ligne, ou commence par non-alphanumérique
                if len(text) < 2 or re.search(r'[\n\r;]', text) or re.match(r'^[^\w]', text):
                    filtered_count += 1
                    continue
                
                # Utiliser (text, label) comme clé pour compter les occurrences
                key = (text, label)
                entity_counter[key] += 1
                entity_types[key] = label
    
    print(f"Export de {len(entity_counter)} entités uniques vers CSV...")
    if filtered_count > 0:
        print(f"  ({filtered_count} entités filtrées pour bruit de formatage)")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['entity', 'type', 'frequency'])
        
        # Trier par fréquence décroissante
        for (text, label), freq in entity_counter.most_common():
            writer.writerow([text, label, freq])
    
    print(f"[OK] {len(entity_counter)} entités exportées vers {output_file}")


def parse_job_title_relation(rel_text: str, entities: list) -> tuple:
    """
    Parse une relation JOB_TITLE pour extraire le titre et le nom de la personne.
    
    Args:
        rel_text: Texte de la relation (ex: "General Douglas MacArthur")
        entities: Liste des entités du document
    
    Returns:
        Tuple (person_name, job_title) ou (None, rel_text) si parsing échoue
    """
    import re
    
    # Pattern 1: "Title Name" (ex: "General Douglas MacArthur")
    # Pattern 2: "Title Name1 Name2" (ex: "Governor James Sevier Conway")
    # Pattern 3: "Name, Title" (ex: "John Doe, Governor")
    
    # Essayer de trouver une entité PERSON dans le texte
    person_entities = [e for e in entities if e.get('label') == 'PERSON']
    
    # Chercher si une entité PERSON est contenue dans rel_text
    for person_ent in person_entities:
        person_name = person_ent.get('text', '')
        if person_name and person_name in rel_text:
            # Extraire le titre (tout sauf le nom de la personne)
            title = rel_text.replace(person_name, '').strip()
            # Nettoyer (enlever virgules, espaces multiples)
            title = re.sub(r'[,\s]+', ' ', title).strip()
            if title:
                return (person_name, title)
    
    # Si pas d'entité trouvée, essayer de parser le format "Title Name"
    # Chercher un pattern: mot(s) capitalisés suivis d'un nom propre
    words = rel_text.split()
    if len(words) >= 2:
        # Les titres sont généralement au début: "General", "Governor", "Captain", etc.
        # Les noms propres suivent
        common_titles = ['General', 'Governor', 'Captain', 'Major', 'Lieutenant', 
                        'Colonel', 'Col', 'Lt', 'Maj', 'Gen', 'Gov', 'attorney',
                        'President', 'Minister', 'Secretary', 'Director']
        
        # Chercher un titre au début
        for i, word in enumerate(words):
            # Nettoyer le mot (enlever ponctuation)
            clean_word = re.sub(r'[.,;:!?]+', '', word)
            if clean_word in common_titles or clean_word.title() in common_titles:
                # Le titre est tout jusqu'à i+1, le reste est le nom
                title = ' '.join(words[:i+1])
                person_name = ' '.join(words[i+1:])
                if person_name:
                    return (person_name, title)
        
        # Si pas de titre reconnu, supposer que le premier mot est le titre
        # et le reste est le nom (approximation)
        if len(words) >= 2:
            title = words[0]
            person_name = ' '.join(words[1:])
            return (person_name, title)
    
    return (None, rel_text)


def export_relations_to_csv(input_file: str, output_file: str):
    """
    Exporte les relations vers un fichier CSV.
    Format: entity1;relation;entity2
    """
    print("Collecte des relations...")
    relations = []
    
    for doc in tqdm(load_ner_data(input_file), desc="Lecture des documents"):
        relations_data = doc.get('relations', [])
        entities = doc.get('entities', [])
        
        for rel in relations_data:
            rel_type = rel.get('type', 'UNKNOWN')
            rel_text = rel.get('text', '')
            
            # Normaliser les anciens noms vers les nouveaux
            if rel_type in ['JOB_TITLE', 'OCCUPATION', 'JOB_TITLE_APPOSITION', 'OCCUPATION_APPOSITION']:
                rel_type = 'OCCUPATION'
            
            if rel_type == 'OCCUPATION':
                # Filtrer les relations invalides
                if not rel_text or len(rel_text.strip()) < 3:
                    continue
                # Rejeter si contient des caractères bizarres ou mots exclus
                excluded = ['>', '<', 'unk', 'unknown', '|', '&']
                if any(exc in rel_text.lower() for exc in excluded):
                    continue
                # Rejeter si trop court après nettoyage
                import re
                cleaned = re.sub(r'[^\w\s]', '', rel_text).strip()
                if len(cleaned) < 3:
                    continue
                
                # Vérifier si person_entity est déjà extraite dans la relation
                person_name = rel.get('person_entity')
                
                if person_name:
                    # Extraire le titre (gère les deux formats: "Title Name" et "Name, Title")
                    if ',' in rel_text:
                        # Format: "Name, Title" - le titre est après la virgule
                        parts = rel_text.split(',', 1)
                        if len(parts) == 2 and person_name in parts[0]:
                            job_title = parts[1].strip()
                        else:
                            job_title = rel_text.replace(person_name, '').strip(',').strip()
                    else:
                        # Format "Title Name" - le titre est avant le nom
                        job_title = rel_text.replace(person_name, '').strip()
                    
                    # Nettoyer (enlever virgules, espaces multiples)
                    job_title = re.sub(r'[,\s]+', ' ', job_title).strip()
                    # Enlever les virgules en début/fin
                    job_title = job_title.strip(',').strip()
                    
                    if not job_title or len(job_title) < 2:
                        continue  # Rejeter si pas de titre valide
                else:
                    # Parser la relation pour extraire personne et titre
                    person_name, job_title = parse_job_title_relation(rel_text, entities)
                
                if person_name and job_title:
                    # Validation finale
                    if len(person_name) >= 2 and len(job_title) >= 2:
                        # Filtrer les relations avec artefacts de tokenisation
                        if not (person_name.strip().endswith("'s") or job_title.strip().endswith("'s")):
                            if not (re.match(r'^[.,;:]', person_name.strip()) or re.match(r'^[.,;:]', job_title.strip())):
                                relations.append({
                                    'entity1': person_name,
                                    'relation': rel_type,
                                    'entity2': job_title
                                })
                else:
                    # Si parsing échoue complètement, rejeter plutôt que mettre UNKNOWN
                    # (on préfère ne pas exporter de mauvaises relations)
                    continue
            elif rel_type == 'TAXONOMY':
                # Pour TAXONOMY, essayer d'extraire les entités du pattern
                # Format: "X such as Y" ou "X like Y"
                # Chercher les entités GPE/ORG/LOC dans le texte
                taxonomy_entities = [e for e in entities if e.get('label') in ['GPE', 'ORG', 'LOC']]
                if taxonomy_entities:
                    # Prendre la première entité trouvée comme entity2
                    entity2 = taxonomy_entities[0].get('text', rel_text)
                    relations.append({
                        'entity1': 'UNKNOWN',  # Le concept général (X)
                        'relation': rel_type,
                        'entity2': entity2  # L'entité spécifique (Y)
                    })
                else:
                    relations.append({
                        'entity1': 'UNKNOWN',
                        'relation': rel_type,
                        'entity2': rel_text
                    })
            elif rel_type in ['SOCIAL_ISSUE', 'SOCIAL_ISSUE_CONFLICT', 'SOCIAL_ISSUE_CRISIS', 'SOCIAL_ISSUE_ACTION']:
                # Relations sociales: entity2 contient le problème social
                relations.append({
                    'entity1': 'UNKNOWN',
                    'relation': rel_type,
                    'entity2': rel_text
                })
            elif rel_type in ['LOCATION_BORN', 'LOCATION_BASED']:
                # Extraire PERSON/ORG et GPE du texte
                person_org = [e for e in entities if e.get('label') in ['PERSON', 'ORG']]
                location = [e for e in entities if e.get('label') in ['GPE', 'LOC', 'FAC']]
                
                if person_org and location:
                    relations.append({
                        'entity1': person_org[0].get('text', 'UNKNOWN'),
                        'relation': rel_type,
                        'entity2': location[0].get('text', 'UNKNOWN')
                    })
                else:
                    # Si pas d'entités trouvées, rejeter
                    continue
            elif rel_type == 'ORG_MEMBER':
                # Extraire PERSON et ORG
                person = [e for e in entities if e.get('label') == 'PERSON']
                org = [e for e in entities if e.get('label') == 'ORG']
                
                if person and org:
                    relations.append({
                        'entity1': person[0].get('text', 'UNKNOWN'),
                        'relation': rel_type,
                        'entity2': org[0].get('text', 'UNKNOWN')
                    })
                else:
                    continue
            elif rel_type == 'EVENT_DATE':
                # Extraire EVENT et DATE
                event = [e for e in entities if e.get('label') == 'EVENT']
                date = [e for e in entities if e.get('label') == 'DATE']
                
                if event and date:
                    event_text = event[0].get('text', '')
                    date_text = date[0].get('text', '').strip()
                    # Valider le format de la date (année 4 chiffres ou format date)
                    import re
                    if re.match(r'^(\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})$', date_text):
                        relations.append({
                            'entity1': event_text,
                            'relation': rel_type,
                            'entity2': date_text
                        })
                # Rejeter les relations EVENT_DATE sans format de date valide
            elif rel_type == 'GEOGRAPHIC_RELATION':
                # Extraire deux GPE/LOC
                locations = [e for e in entities if e.get('label') in ['GPE', 'LOC']]
                
                if len(locations) >= 2:
                    relations.append({
                        'entity1': locations[0].get('text', 'UNKNOWN'),
                        'relation': rel_type,
                        'entity2': locations[1].get('text', 'UNKNOWN')
                    })
                else:
                    continue
            else:
                # Relations non gérées: rejeter plutôt que mettre UNKNOWN
                continue
        
        # Filtrer les relations avec entity1 = "UNKNOWN" (sauf pour certains types spéciaux)
        filtered_relations = []
        for rel in relations:
            # Garder seulement les relations où entity1 n'est pas UNKNOWN (sauf pour certains types spéciaux)
            if rel['entity1'] != 'UNKNOWN' or rel['relation'] in ['TAXONOMY', 'SOCIAL_ISSUE_CONFLICT', 'SOCIAL_ISSUE_CRISIS', 'SOCIAL_ISSUE_ACTION']:
                # Pour les types spéciaux, vérifier que entity2 est valide
                if rel['entity2'] != 'UNKNOWN':
                    filtered_relations.append(rel)
        
        relations = filtered_relations
    
    print(f"Export de {len(relations)} relations vers CSV...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['entity1', 'relation', 'entity2'])
        
        for rel in relations:
            writer.writerow([
                rel['entity1'],
                rel['relation'],
                rel['entity2']
            ])
    
    print(f"[OK] {len(relations)} relations exportées vers {output_file}")


def export_acronyms_to_csv(input_file: str, output_file: str):
    """
    Exporte les acronymes vers un fichier CSV.
    Format: acronym;full_form
    """
    import re
    
    print("Collecte des acronymes...")
    acronyms_dict = {}
    acronym_counts = Counter()  # Compter les occurrences pour filtrer les faux positifs
    
    # Mots à exclure (initiales simples, nombres seuls, etc.)
    excluded_patterns = [
        r'^[A-Z]\.$',  # Une seule lettre avec point (ex: "B.")
        r'^[IVX]+$',   # Nombres romains seuls (ex: "II", "IV")
        r'^[0-9]+$',   # Nombres seuls
        r'^[A-Z]\.?$', # Une seule lettre (ex: "B", "B.")
    ]
    
    for doc in tqdm(load_ner_data(input_file), desc="Lecture des documents"):
        acronyms = doc.get('acronyms', [])
        
        for acro in acronyms:
            acronym = acro.get('acronym', '').strip()
            definition = acro.get('definition', 'unknown').strip()
            
            if not acronym:
                continue
            
            # Filtrer les initiales simples et autres faux positifs
            is_excluded = False
            for pattern in excluded_patterns:
                if re.match(pattern, acronym):
                    is_excluded = True
                    break
            
            if is_excluded:
                continue
            
            # Filtrer les acronymes trop courts (moins de 2 caractères sans le point)
            clean_acronym = acronym.replace('.', '')
            if len(clean_acronym) < 2:
                continue
            
            # Compter les occurrences
            acronym_counts[acronym] += 1
            
            # Garder la meilleure définition si plusieurs occurrences
            if acronym not in acronyms_dict:
                acronyms_dict[acronym] = definition
            elif definition != 'unknown' and acronyms_dict[acronym] == 'unknown':
                # Remplacer "unknown" par une vraie définition si trouvée
                acronyms_dict[acronym] = definition
            elif definition != 'unknown' and acronyms_dict[acronym] != 'unknown':
                # Si les deux ont des définitions, garder la plus courte (généralement plus précise)
                if len(definition) < len(acronyms_dict[acronym]):
                    acronyms_dict[acronym] = definition
    
    # Note: On garde tous les acronymes qui ont passé les filtres de qualité
    # (initiales simples, nombres, etc.). Le filtre de fréquence est retiré car
    # pour de petits échantillons, il est normal qu'un acronyme n'apparaisse qu'une fois.
    filtered_acronyms = acronyms_dict.copy()
    
    print(f"Export de {len(filtered_acronyms)} acronymes uniques vers CSV...")
    print(f"  (dont {sum(1 for d in filtered_acronyms.values() if d != 'unknown')} avec définition connue)")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['acronym', 'full_form'])
        
        # Trier par: (1) définition connue d'abord, (2) puis par fréquence, (3) puis alphabétiquement
        sorted_items = sorted(
            filtered_acronyms.items(),
            key=lambda x: (
                x[1] == 'unknown',  # "unknown" en dernier
                -acronym_counts[x[0]],  # Plus fréquents d'abord
                x[0]  # Alphabétiquement
            )
        )
        
        for acronym, full_form in sorted_items:
            writer.writerow([acronym, full_form])
    
    print(f"[OK] {len(filtered_acronyms)} acronymes exportés vers {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Export des résultats vers CSV')
    parser.add_argument('--entities-input', type=str,
                       default='results/normalized/entities_normalized.jsonl',
                       help='Fichier JSONL avec entités normalisées')
    parser.add_argument('--relations-input', type=str,
                       default='results/raw/relations.jsonl',
                       help='Fichier JSONL avec relations')
    parser.add_argument('--output-dir', type=str,
                       default='results/csv',
                       help='Répertoire de sortie pour les fichiers CSV')
    
    args = parser.parse_args()
    
    print("="*80)
    print("EXPORT VERS CSV")
    print("="*80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Export des entités
    if os.path.exists(args.entities_input):
        print("\n1. Export des entités...")
        export_entities_to_csv(
            args.entities_input,
            os.path.join(args.output_dir, 'entities.csv')
        )
    else:
        print(f"\n⚠️  Fichier d'entités non trouvé: {args.entities_input}")
    
    # 2. Export des relations
    if os.path.exists(args.relations_input):
        print("\n2. Export des relations...")
        export_relations_to_csv(
            args.relations_input,
            os.path.join(args.output_dir, 'relations.csv')
        )
    else:
        print(f"\n⚠️  Fichier de relations non trouvé: {args.relations_input}")
    
    # 3. Export des acronymes (depuis le fichier de relations qui contient aussi les acronymes)
    if os.path.exists(args.relations_input):
        print("\n3. Export des acronymes...")
        export_acronyms_to_csv(
            args.relations_input,
            os.path.join(args.output_dir, 'acronyms.csv')
        )
    else:
        print(f"\n⚠️  Fichier de relations non trouvé: {args.relations_input}")
    
    print("\n" + "="*80)
    print("[OK] EXPORT TERMINÉ")
    print("="*80)
    print(f"Fichiers CSV disponibles dans: {args.output_dir}")
    print("  - entities.csv")
    print("  - relations.csv")
    print("  - acronyms.csv")


if __name__ == '__main__':
    main()

