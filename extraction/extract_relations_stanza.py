"""
Étape 5 (Alternative): Extraction des relations avec Stanza.
Utilise Stanza au lieu de spaCy pour un meilleur parsing de dépendances.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from tqdm import tqdm
from utils.data_loader import load_ner_data
from utils.information_extractor_stanza import InformationExtractorStanza


def extract_relations_with_stanza(input_file: str, output_file: str, max_docs: int = None):
    """
    Extrait les relations depuis un fichier JSONL en utilisant Stanza.
    
    Args:
        input_file: Fichier JSONL avec entités (doit contenir 'text')
        output_file: Fichier JSONL de sortie avec relations
        max_docs: Nombre maximum de documents à traiter
    """
    print("Initialisation de l'extracteur Stanza...")
    extractor = InformationExtractorStanza()
    
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
    
    print("Extraction des relations avec Stanza...")
    if max_docs:
        print(f"Limite: {max_docs:,} documents")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_data in tqdm(load_ner_data(input_file), desc="Traitement des documents"):
            if max_docs and stats['total_docs'] >= max_docs:
                break
            
            stats['total_docs'] += 1
            
            # Récupérer le texte du document
            if 'text' not in doc_data:
                continue
            
            text = doc_data['text']
            
            # Extraire les relations
            relations = extractor.extract_relations(text)
            
            # Extraire les acronymes
            acronyms = extractor.extract_acronyms(text)
            
            # Extraire les entités normalisées
            normalized_entities = extractor.normalize_entities(text)
            
            # Sauvegarder seulement si on a trouvé quelque chose
            if relations or acronyms:
                record = {
                    'text_snippet': text[:100] + "..." if len(text) > 100 else text,
                    'relations': relations,
                    'acronyms': acronyms,
                }
                
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
    print("STATISTIQUES D'EXTRACTION (STANZA)")
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
    stats['acronyms_found'] = list(stats['acronyms_found'])
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nStatistiques sauvegardées dans: {stats_file}")
    print(f"Relations sauvegardées dans: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extraction des relations avec Stanza'
    )
    parser.add_argument('--input', type=str, 
                       default='results/normalized/entities_normalized.jsonl',
                       help='Fichier JSONL d\'entrée avec entités')
    parser.add_argument('--output', type=str,
                       default='results/raw/relations_stanza.jsonl',
                       help='Fichier JSONL de sortie avec relations')
    parser.add_argument('--max-docs', type=int, default=None,
                       help='Nombre maximum de documents à traiter')
    
    args = parser.parse_args()
    
    extract_relations_with_stanza(
        args.input,
        args.output,
        max_docs=args.max_docs
    )

