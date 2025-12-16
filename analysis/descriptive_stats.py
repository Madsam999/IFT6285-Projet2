"""
Script pour l'analyse descriptive des entités nommées extraites.
Implémente les sections 1.1, 1.2 et 1.3 du plan.
"""
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import sys
import os

# Ajouter le répertoire parent au path pour importer utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from utils.data_loader import load_ner_data


def calculate_general_stats(data: List[Dict]) -> Dict:
    """
    Calcule les statistiques générales (Section 1.1).
    
    Args:
        data: Liste de dictionnaires avec 'text' et 'entities'
        
    Returns:
        Dictionnaire avec les statistiques
    """
    total_docs = len(data)
    total_entities = sum(len(doc['entities']) for doc in data)
    docs_with_entities = sum(1 for doc in data if len(doc['entities']) > 0)
    docs_without_entities = total_docs - docs_with_entities
    avg_entities_per_doc = total_entities / total_docs if total_docs > 0 else 0
    
    # Distribution par type d'entité
    entity_type_dist = Counter()
    for doc in data:
        for entity in doc['entities']:
            entity_type_dist[entity['label']] += 1
    
    return {
        'total_documents': total_docs,
        'total_entities': total_entities,
        'documents_with_entities': docs_with_entities,
        'documents_without_entities': docs_without_entities,
        'avg_entities_per_doc': round(avg_entities_per_doc, 2),
        'entity_type_distribution': dict(entity_type_dist)
    }


def analyze_by_entity_type(data: List[Dict]) -> Dict:
    """
    Analyse par type d'entité (Section 1.2).
    
    Args:
        data: Liste de dictionnaires avec 'text' et 'entities'
        
    Returns:
        Dictionnaire avec les statistiques par type
    """
    type_stats = defaultdict(lambda: {
        'count': 0,
        'entities': Counter(),
        'lengths': []
    })
    
    for doc in data:
        for entity in doc['entities']:
            label = entity['label']
            text = entity['text']
            
            type_stats[label]['count'] += 1
            type_stats[label]['entities'][text.lower()] += 1
            type_stats[label]['lengths'].append(len(text))
    
    # Calculer les moyennes et préparer les résultats
    results = {}
    for label, stats in type_stats.items():
        results[label] = {
            'total_occurrences': stats['count'],
            'unique_entities': len(stats['entities']),
            'avg_length': round(sum(stats['lengths']) / len(stats['lengths']), 2) if stats['lengths'] else 0,
            'most_frequent': dict(stats['entities'].most_common(10))
        }
    
    return results


def analyze_most_frequent_entities(data: List[Dict]) -> Dict:
    """
    Analyse des entités les plus fréquentes (Section 1.3).
    
    Args:
        data: Liste de dictionnaires avec 'text' et 'entities'
        
    Returns:
        Dictionnaire avec les entités les plus fréquentes
    """
    # Toutes les entités (tous types confondus)
    all_entities = Counter()
    
    # Par type
    entities_by_type = defaultdict(Counter)
    
    # Co-occurrences dans les mêmes documents
    co_occurrences = defaultdict(int)
    
    for doc in data:
        doc_entities = []
        for entity in doc['entities']:
            text = entity['text'].lower()
            label = entity['label']
            
            all_entities[text] += 1
            entities_by_type[label][text] += 1
            doc_entities.append((text, label))
        
        # Calculer les co-occurrences dans ce document
        for i, (ent1, label1) in enumerate(doc_entities):
            for j, (ent2, label2) in enumerate(doc_entities[i+1:], start=i+1):
                # Créer une clé symétrique pour les co-occurrences
                pair = tuple(sorted([(ent1, label1), (ent2, label2)]))
                co_occurrences[pair] += 1
    
    # Top co-occurrences
    top_co_occurrences = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)[:20]
    
    return {
        'most_frequent_all_types': dict(all_entities.most_common(30)),
        'most_frequent_by_type': {label: dict(counter.most_common(10)) 
                                  for label, counter in entities_by_type.items()},
        'top_co_occurrences': [(f"{pair[0][0]} ({pair[0][1]}) & {pair[1][0]} ({pair[1][1]})", count)
                               for pair, count in top_co_occurrences]
    }


def save_stats(stats: Dict, filepath: str):
    """Sauvegarde les statistiques dans un fichier JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def print_stats(general_stats: Dict, type_stats: Dict, frequent_stats: Dict):
    """Affiche les statistiques de manière lisible."""
    print("\n" + "="*80)
    print("STATISTIQUES GÉNÉRALES (Section 1.1)")
    print("="*80)
    print(f"Nombre total de documents/chunks: {general_stats['total_documents']:,}")
    print(f"Nombre total d'entités extraites: {general_stats['total_entities']:,}")
    print(f"Documents avec entités: {general_stats['documents_with_entities']:,}")
    print(f"Documents sans entités: {general_stats['documents_without_entities']:,}")
    print(f"Fréquence moyenne d'entités par document: {general_stats['avg_entities_per_doc']:.2f}")
    
    print("\nDistribution des entités par type:")
    for label, count in sorted(general_stats['entity_type_distribution'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count:,}")
    
    print("\n" + "="*80)
    print("ANALYSE PAR TYPE D'ENTITÉ (Section 1.2)")
    print("="*80)
    for label, stats in sorted(type_stats.items()):
        print(f"\n{label}:")
        print(f"  Total d'occurrences: {stats['total_occurrences']:,}")
        print(f"  Entités uniques: {stats['unique_entities']:,}")
        print(f"  Longueur moyenne: {stats['avg_length']:.2f} caractères")
        print(f"  Top 5 entités les plus fréquentes:")
        for entity, count in list(stats['most_frequent'].items())[:5]:
            print(f"    - {entity}: {count}")
    
    print("\n" + "="*80)
    print("ENTITÉS LES PLUS FRÉQUENTES (Section 1.3)")
    print("="*80)
    print("\nTop 20 entités (tous types confondus):")
    for i, (entity, count) in enumerate(list(frequent_stats['most_frequent_all_types'].items())[:20], 1):
        print(f"  {i:2d}. {entity}: {count}")
    
    print("\nTop 10 entités par type:")
    for label, entities in frequent_stats['most_frequent_by_type'].items():
        print(f"\n  {label}:")
        for entity, count in list(entities.items())[:10]:
            print(f"    - {entity}: {count}")
    
    print("\n" + "="*80)
    print("TOP 10 CO-OCCURRENCES")
    print("="*80)
    for i, (pair, count) in enumerate(frequent_stats['top_co_occurrences'][:10], 1):
        print(f"  {i:2d}. {pair}: {count}")


def calculate_stats_streaming(filepath: str, max_docs: int = None, compute_cooccurrences: bool = True, max_entities_for_cooccur: int = 20) -> Tuple[Dict, Dict, Dict]:
    """
    Calcule toutes les statistiques en mode streaming pour économiser la mémoire.
    
    Args:
        filepath: Chemin vers le fichier JSONL
        max_docs: Nombre maximum de documents à traiter (None = tous)
        compute_cooccurrences: Si False, désactive le calcul des co-occurrences (beaucoup plus rapide)
        max_entities_for_cooccur: Nombre max d'entités par document pour calculer les co-occurrences
    """
    from utils.data_loader import load_ner_data
    
    # Compteurs pour les statistiques générales
    total_docs = 0
    total_entities = 0
    docs_with_entities = 0
    entity_type_dist = Counter()
    
    # Compteurs pour l'analyse par type
    type_stats = defaultdict(lambda: {
        'count': 0,
        'entities': Counter(),
        'lengths': []
    })
    
    # Compteurs pour les entités les plus fréquentes
    all_entities = Counter()
    entities_by_type = defaultdict(Counter)
    co_occurrences = defaultdict(int)
    
    print("Traitement des données en streaming...")
    if max_docs:
        print(f"Limite: {max_docs:,} documents")
    if not compute_cooccurrences:
        print("Co-occurrences désactivées (mode rapide)")
    
    for doc in tqdm(load_ner_data(filepath), desc="Analyse des documents"):
        if max_docs and total_docs >= max_docs:
            break
            
        total_docs += 1
        num_entities = len(doc['entities'])
        total_entities += num_entities
        
        if num_entities > 0:
            docs_with_entities += 1
        
        doc_entities = []
        for entity in doc['entities']:
            label = entity['label']
            text = entity['text'].lower()
            
            # Statistiques générales
            entity_type_dist[label] += 1
            
            # Analyse par type
            type_stats[label]['count'] += 1
            type_stats[label]['entities'][text] += 1
            type_stats[label]['lengths'].append(len(text))
            
            # Entités les plus fréquentes
            all_entities[text] += 1
            entities_by_type[label][text] += 1
            doc_entities.append((text, label))
        
        # Co-occurrences (seulement si activé et document pas trop grand)
        if compute_cooccurrences and len(doc_entities) <= max_entities_for_cooccur:
            for i, (ent1, label1) in enumerate(doc_entities):
                for j, (ent2, label2) in enumerate(doc_entities[i+1:], start=i+1):
                    pair = tuple(sorted([(ent1, label1), (ent2, label2)]))
                    co_occurrences[pair] += 1
    
    # Construire les résultats
    avg_entities_per_doc = total_entities / total_docs if total_docs > 0 else 0
    general_stats = {
        'total_documents': total_docs,
        'total_entities': total_entities,
        'documents_with_entities': docs_with_entities,
        'documents_without_entities': total_docs - docs_with_entities,
        'avg_entities_per_doc': round(avg_entities_per_doc, 2),
        'entity_type_distribution': dict(entity_type_dist)
    }
    
    # Préparer les statistiques par type
    type_results = {}
    for label, stats in type_stats.items():
        type_results[label] = {
            'total_occurrences': stats['count'],
            'unique_entities': len(stats['entities']),
            'avg_length': round(sum(stats['lengths']) / len(stats['lengths']), 2) if stats['lengths'] else 0,
            'most_frequent': dict(stats['entities'].most_common(10))
        }
    
    # Préparer les entités les plus fréquentes
    top_co_occurrences = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)[:20]
    frequent_stats = {
        'most_frequent_all_types': dict(all_entities.most_common(30)),
        'most_frequent_by_type': {label: dict(counter.most_common(10)) 
                                  for label, counter in entities_by_type.items()},
        'top_co_occurrences': [(f"{pair[0][0]} ({pair[0][1]}) & {pair[1][0]} ({pair[1][1]})", count)
                               for pair, count in top_co_occurrences]
    }
    
    return general_stats, type_results, frequent_stats


def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyse descriptive des entités NER')
    parser.add_argument('--max-docs', type=int, default=None, 
                       help='Nombre maximum de documents à traiter (pour test rapide)')
    parser.add_argument('--no-cooccurrences', action='store_true',
                       help='Désactive le calcul des co-occurrences (beaucoup plus rapide)')
    parser.add_argument('--input', type=str, default='ner_results_sample.jsonl',
                       help='Fichier d\'entrée JSONL')
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = "results/stats/descriptive_stats.json"
    
    if not os.path.exists(input_file):
        print(f"Erreur: Le fichier {input_file} n'existe pas.")
        return
    
    print(f"Analyse des données depuis {input_file}...")
    general_stats, type_stats, frequent_stats = calculate_stats_streaming(
        input_file, 
        max_docs=args.max_docs,
        compute_cooccurrences=not args.no_cooccurrences
    )
    
    # Combiner toutes les statistiques
    all_stats = {
        'general_stats': general_stats,
        'type_stats': type_stats,
        'frequent_stats': frequent_stats
    }
    
    # Sauvegarder
    save_stats(all_stats, output_file)
    print(f"\nStatistiques sauvegardées dans {output_file}")
    
    # Afficher
    print_stats(general_stats, type_stats, frequent_stats)


if __name__ == "__main__":
    main()

