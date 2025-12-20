"""
Étape 3: Réduction du bruit dans les entités extraites.
Applique la normalisation basée sur l'analyse des variantes.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from collections import defaultdict
from tqdm import tqdm
from utils.data_loader import load_ner_data


def load_variant_mapping(variant_analysis_file: str) -> dict:
    """
    Charge l'analyse des variantes et crée un mapping variante -> entité canonique.
    
    Returns:
        Dictionnaire {variant_text: canonical_text} pour chaque type d'entité
    """
    with open(variant_analysis_file, 'r', encoding='utf-8') as f:
        variant_stats = json.load(f)
    
    mapping = {}
    
    for label, stats in variant_stats['variant_stats_by_type'].items():
        # Utiliser 'all_variant_clusters' si disponible (tous les clusters), sinon 'top_variant_clusters' (rétrocompatibilité)
        clusters_to_use = stats.get('all_variant_clusters', stats.get('top_variant_clusters', []))
        for cluster in clusters_to_use:
            canonical = cluster['canonical']
            for variant in cluster['variants']:
                variant_text = variant['text']
                # Créer une clé unique par type d'entité
                key = (variant_text, label)
                mapping[key] = canonical
    
    return mapping


def normalize_entities(input_file: str, output_file: str, 
                     variant_mapping: dict = None,
                     variant_analysis_file: str = None):
    """
    Normalise les entités en appliquant le mapping des variantes.
    
    Args:
        input_file: Fichier JSONL d'entrée avec entités brutes
        output_file: Fichier JSONL de sortie avec entités normalisées
        variant_mapping: Mapping direct (optionnel)
        variant_analysis_file: Fichier d'analyse des variantes (optionnel)
    """
    if variant_mapping is None and variant_analysis_file:
        print("Chargement du mapping des variantes...")
        variant_mapping = load_variant_mapping(variant_analysis_file)
        print(f"  {len(variant_mapping)} mappings chargés")
    
    if variant_mapping is None:
        print("Avertissement: Aucun mapping de variantes fourni. Les entités ne seront pas normalisées.")
        variant_mapping = {}
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    stats = {
        'total_entities': 0,
        'normalized_entities': 0,
        'normalizations_by_type': defaultdict(int)
    }
    
    print("Normalisation des entités...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in tqdm(load_ner_data(input_file), desc="Traitement des documents"):
            normalized_entities = []
            
            for entity in doc['entities']:
                stats['total_entities'] += 1
                original_text = entity['text']
                label = entity['label']
                
                # Chercher dans le mapping
                key = (original_text, label)
                if key in variant_mapping:
                    # Normaliser vers l'entité canonique
                    canonical = variant_mapping[key]
                    normalized_entity = entity.copy()
                    normalized_entity['text'] = canonical
                    normalized_entity['original_text'] = original_text  # Garder l'original pour référence
                    normalized_entities.append(normalized_entity)
                    stats['normalized_entities'] += 1
                    stats['normalizations_by_type'][label] += 1
                else:
                    # Pas de normalisation nécessaire
                    normalized_entities.append(entity)
            
            # Sauvegarder le document avec entités normalisées
            record = {"entities": normalized_entities}
            if 'text' in doc:
                record["text"] = doc['text']
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Afficher les statistiques
    print("\n" + "="*80)
    print("STATISTIQUES DE NORMALISATION")
    print("="*80)
    print(f"Total d'entités traitées: {stats['total_entities']:,}")
    print(f"Entités normalisées: {stats['normalized_entities']:,} ({stats['normalized_entities']/stats['total_entities']*100:.2f}%)")
    print("\nNormalisations par type:")
    for label, count in sorted(stats['normalizations_by_type'].items(), 
                              key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count:,}")
    
    # Sauvegarder les stats
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStatistiques sauvegardées dans: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Étape 3: Réduction du bruit dans les entités')
    parser.add_argument('--input', type=str, 
                       default='results/raw/entities_raw.jsonl',
                       help='Fichier d\'entrée avec entités brutes')
    parser.add_argument('--output', type=str,
                       default='results/normalized/entities_normalized.jsonl',
                       help='Fichier de sortie avec entités normalisées')
    parser.add_argument('--variant-analysis', type=str,
                       default='results/stats/variant_analysis.json',
                       help='Fichier d\'analyse des variantes (étape 2)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Erreur: Le fichier {args.input} n'existe pas.")
        print("Veuillez d'abord exécuter extraction/extract_entities.py")
        return
    
    normalize_entities(
        args.input,
        args.output,
        variant_analysis_file=args.variant_analysis
    )
    
    print(f"\n✓ Normalisation terminée. Résultats sauvegardés dans: {args.output}")


if __name__ == '__main__':
    main()

