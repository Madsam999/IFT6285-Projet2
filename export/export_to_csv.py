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
    """
    print("Collecte des entités...")
    entity_counter = Counter()
    entity_types = {}
    
    for doc in tqdm(load_ner_data(input_file), desc="Lecture des documents"):
        for entity in doc.get('entities', []):
            text = entity.get('text', '')
            label = entity.get('label', 'UNKNOWN')
            
            if text:
                # Utiliser (text, label) comme clé pour compter les occurrences
                key = (text, label)
                entity_counter[key] += 1
                entity_types[key] = label
    
    print(f"Export de {len(entity_counter)} entités uniques vers CSV...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['entity', 'type', 'frequency'])
        
        # Trier par fréquence décroissante
        for (text, label), freq in entity_counter.most_common():
            writer.writerow([text, label, freq])
    
    print(f"✓ {len(entity_counter)} entités exportées vers {output_file}")


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
            if rel_type in ['JOB_TITLE', 'OCCUPATION']:
                rel_type = 'OCCUPATION'
            elif rel_type in ['JOB_TITLE_APPOSITION', 'OCCUPATION_APPOSITION']:
                rel_type = 'OCCUPATION_APPOSITION'
            
            if rel_type in ['OCCUPATION', 'OCCUPATION_APPOSITION']:
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
                    # Extraire le titre selon le type de relation
                    if rel_type == 'OCCUPATION_APPOSITION':
                        # Format: "Name, Title" - le titre est après la virgule
                        if ',' in rel_text:
                            parts = rel_text.split(',', 1)
                            if len(parts) == 2:
                                # Vérifier que la première partie contient le nom
                                if person_name in parts[0]:
                                    job_title = parts[1].strip()
                                else:
                                    # Fallback: enlever le nom du texte complet
                                    job_title = rel_text.replace(person_name, '').strip(',').strip()
                            else:
                                job_title = rel_text.replace(person_name, '').strip(',').strip()
                        else:
                            # Pas de virgule, enlever le nom
                            job_title = rel_text.replace(person_name, '').strip(',').strip()
                    else:
                        # OCCUPATION: Format "Title Name" - le titre est avant le nom
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
                # Pour TAXONOMY, le format est généralement "X IS_A Y"
                # entity2 contient l'entité, entity1 serait le concept
                # Pour l'instant, on met le texte dans entity2
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
            else:
                relations.append({
                    'entity1': 'UNKNOWN',
                    'relation': rel_type,
                    'entity2': rel_text
                })
    
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
    
    print(f"✓ {len(relations)} relations exportées vers {output_file}")


def export_acronyms_to_csv(input_file: str, output_file: str):
    """
    Exporte les acronymes vers un fichier CSV.
    Format: acronym;full_form
    """
    print("Collecte des acronymes...")
    acronyms_dict = {}
    
    for doc in tqdm(load_ner_data(input_file), desc="Lecture des documents"):
        acronyms = doc.get('acronyms', [])
        
        for acro in acronyms:
            acronym = acro.get('acronym', '')
            definition = acro.get('definition', 'unknown')
            
            if acronym:
                # Garder la meilleure définition si plusieurs occurrences
                if acronym not in acronyms_dict or definition != 'unknown':
                    if acronyms_dict.get(acronym) == 'unknown' or definition != 'unknown':
                        acronyms_dict[acronym] = definition
    
    print(f"Export de {len(acronyms_dict)} acronymes uniques vers CSV...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['acronym', 'full_form'])
        
        for acronym, full_form in sorted(acronyms_dict.items()):
            writer.writerow([acronym, full_form])
    
    print(f"✓ {len(acronyms_dict)} acronymes exportés vers {output_file}")


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
    print("✓ EXPORT TERMINÉ")
    print("="*80)
    print(f"Fichiers CSV disponibles dans: {args.output_dir}")
    print("  - entities.csv")
    print("  - relations.csv")
    print("  - acronyms.csv")


if __name__ == '__main__':
    main()

