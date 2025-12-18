"""
Étape 5: Extraction des relations entre entités.
Utilise InformationExtractor pour extraire les relations, acronymes, et autres patterns.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy
import json
import argparse
from tqdm import tqdm
from utils.data_loader import load_ner_data
from utils.information_extractor import InformationExtractor
from main import MODEL_NAME, BATCH_SIZE


def extract_relations_from_jsonl(input_file: str, output_file: str, 
                                 nlp_model_name: str = MODEL_NAME,
                                 max_docs: int = None):
    """
    Extrait les relations depuis un fichier JSONL contenant des entités.
    
    Args:
        input_file: Fichier JSONL avec entités (peut contenir 'text' ou non)
        output_file: Fichier JSONL de sortie avec relations
        nlp_model_name: Nom du modèle spaCy à utiliser
    """
    print(f"Chargement du modèle spaCy: {nlp_model_name}...")
    nlp = spacy.load(nlp_model_name)
    
    # Pour les relations, on a besoin du parser et du tagger
    # Ne pas désactiver ces pipes
    print("Initialisation de l'extracteur d'information...")
    extractor = InformationExtractor(nlp)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    stats = {
        'total_docs': 0,
        'docs_with_relations': 0,
        'docs_with_acronyms': 0,
        'total_relations': 0,
        'total_acronyms': 0,
        'relation_types': {},
        'acronyms_found': set()
    }
    
    print("Extraction des relations...")
    if max_docs:
        print(f"Limite: {max_docs:,} documents")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_data in tqdm(load_ner_data(input_file), desc="Traitement des documents"):
            if max_docs and stats['total_docs'] >= max_docs:
                break
            
            stats['total_docs'] += 1
            
            # Récupérer le texte du document
            # Si le texte n'est pas stocké, on ne peut pas extraire de relations
            if 'text' not in doc_data:
                continue
            
            text = doc_data['text']
            
            # Traiter le texte avec spaCy
            doc = nlp(text)
            
            # Extraire les relations
            relations = extractor.extract_relations(doc)
            
            # Extraire les acronymes
            acronyms = extractor._extract_acronyms(doc)
            
            # Extraire les entités normalisées (pour référence et parsing des relations)
            normalized_entities = extractor.normalize_entities(doc)
            
            # Enrichir les relations OCCUPATION avec les entités PERSON trouvées
            # Cela permettra un meilleur parsing dans l'export CSV
            for rel in relations:
                if rel.get('type') in ['OCCUPATION', 'OCCUPATION_APPOSITION', 'JOB_TITLE', 'JOB_TITLE_APPOSITION']:
                    rel_text = rel.get('text', '')
                    # Filtrer les relations invalides
                    if not rel_text or len(rel_text.strip()) < 3:
                        continue
                    # Rejeter si contient des caractères bizarres
                    if any(char in rel_text for char in ['>', '<', '|', '&', 'unk', 'unknown']):
                        continue
                    
                    # Chercher les entités PERSON dans le texte de la relation
                    for ent in normalized_entities:
                        if ent.get('label') == 'PERSON':
                            person_text = ent.get('text', '')
                            if person_text and person_text in rel_text:
                                rel['person_entity'] = person_text
                                break
            
            # Sauvegarder seulement si on a trouvé quelque chose
            if relations or acronyms:
                record = {
                    'text_snippet': text[:100] + "..." if len(text) > 100 else text,
                    'relations': relations,
                    'acronyms': acronyms,
                }
                
                # Ajouter les entités si disponibles (pour contexte)
                if normalized_entities:
                    record['entities'] = normalized_entities
                
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Mettre à jour les statistiques
                if relations:
                    stats['docs_with_relations'] += 1
                    stats['total_relations'] += len(relations)
                    for rel in relations:
                        rel_type = rel.get('type', 'UNKNOWN')
                        stats['relation_types'][rel_type] = stats['relation_types'].get(rel_type, 0) + 1
                
                if acronyms:
                    stats['docs_with_acronyms'] += 1
                    stats['total_acronyms'] += len(acronyms)
                    for acro in acronyms:
                        acro_text = acro.get('acronym', '')
                        if acro_text:
                            stats['acronyms_found'].add(acro_text)
    
    # Afficher les statistiques
    print("\n" + "="*80)
    print("STATISTIQUES D'EXTRACTION")
    print("="*80)
    print(f"Documents traités: {stats['total_docs']:,}")
    print(f"Documents avec relations: {stats['docs_with_relations']:,}")
    print(f"Documents avec acronymes: {stats['docs_with_acronyms']:,}")
    print(f"Total de relations extraites: {stats['total_relations']:,}")
    print(f"Total d'acronymes extraits: {stats['total_acronyms']:,}")
    print(f"Acronymes uniques: {len(stats['acronyms_found']):,}")
    
    print("\nTypes de relations:")
    for rel_type, count in sorted(stats['relation_types'].items(), 
                                  key=lambda x: x[1], reverse=True):
        print(f"  {rel_type}: {count:,}")
    
    # Sauvegarder les stats
    stats_file = output_file.replace('.jsonl', '_stats.json')
    # Convertir set en list pour JSON
    stats['acronyms_found'] = list(stats['acronyms_found'])
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStatistiques sauvegardées dans: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Étape 5: Extraction des relations entre entités')
    parser.add_argument('--input', type=str,
                       default='results/normalized/entities_normalized.jsonl',
                       help='Fichier JSONL d\'entrée avec entités (doit contenir le texte)')
    parser.add_argument('--output', type=str,
                       default='results/raw/relations.jsonl',
                       help='Fichier JSONL de sortie avec relations')
    parser.add_argument('--model', type=str, default=MODEL_NAME,
                       help='Modèle spaCy à utiliser')
    parser.add_argument('--max-docs', type=int, default=None,
                       help='Nombre maximum de documents à traiter')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Erreur: Le fichier {args.input} n'existe pas.")
        print("Veuillez d'abord exécuter normalization/reduce_noise.py")
        return
    
    print("="*80)
    print("ÉTAPE 5: EXTRACTION DES RELATIONS")
    print("="*80)
    print(f"⚠️  ATTENTION: Cette étape nécessite que les entités contiennent le texte complet.")
    print(f"   Assurez-vous que STORE_TEXT=True lors de l'extraction des entités.")
    print()
    
    # Vérifier que le fichier contient du texte
    sample_checked = False
    for doc in load_ner_data(args.input):
        if 'text' not in doc:
            print("❌ Erreur: Le fichier d'entrée ne contient pas le texte des documents.")
            print("   Les relations nécessitent le texte complet pour fonctionner.")
            print("   Veuillez réexécuter extraction/extract_entities.py avec STORE_TEXT=True")
            return
        if not sample_checked:
            sample_checked = True
            break
    
    extract_relations_from_jsonl(
        args.input,
        args.output,
        nlp_model_name=args.model,
        max_docs=args.max_docs
    )
    
    print(f"\n✓ Extraction terminée. Résultats sauvegardés dans: {args.output}")


if __name__ == '__main__':
    main()

