"""
Analyse du bruit dans les entités extraites (variantes d'entités).
Détecte et quantifie les variantes d'entités (ex: "Olivier Royat", "O. Royan", "M. Royan").
"""
import json
import sys
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
from difflib import SequenceMatcher
import re

# Ajouter le répertoire parent au path pour importer utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from utils.data_loader import load_ner_data

# Seuils de similarité pour détecter les variantes
SIMILARITY_THRESHOLD = 0.9  # Seuil pour considérer deux entités comme variantes
MIN_LENGTH_FOR_VARIANTS = 3  # Longueur minimale pour analyser les variantes
# Note: Le seuil de 0.8 est combiné avec des vérifications de préfixe et de distance d'édition
# pour éviter les faux positifs (ex: "Bollywood" vs "Hollywood" qui ont une similarité élevée
# mais sont des entités différentes car leurs préfixes diffèrent)


def normalize_entity_text(text: str) -> str:
    """
    Normalise le texte d'une entité pour la comparaison.
    - Enlève les espaces multiples
    - Enlève la ponctuation en fin de mot
    - Garde la casse (important pour distinguer "O. Royan" vs "o. royan")
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    # Enlever ponctuation en fin seulement
    text = re.sub(r'[.,;:!?]+$', '', text)
    return text


def string_similarity(s1: str, s2: str) -> float:
    """
    Calcule la similarité entre deux chaînes (0-1).
    Utilise SequenceMatcher de difflib (ratio).
    """
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def edit_distance(s1: str, s2: str) -> int:
    """
    Calcule la distance de Levenshtein (edit distance) entre deux chaînes.
    """
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    
    if len(s1_lower) < len(s2_lower):
        return edit_distance(s2_lower, s1_lower)
    
    if len(s2_lower) == 0:
        return len(s1_lower)
    
    previous_row = range(len(s2_lower) + 1)
    for i, c1 in enumerate(s1_lower):
        current_row = [i + 1]
        for j, c2 in enumerate(s2_lower):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def prefix_similarity(s1: str, s2: str, min_length: int = 3) -> float:
    """
    Calcule la similarité des préfixes (débuts) de deux chaînes.
    Retourne le ratio de caractères identiques au début.
    """
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    min_len = min(len(s1_lower), len(s2_lower), min_length)
    
    if min_len == 0:
        return 0.0
    
    matches = sum(1 for i in range(min_len) if s1_lower[i] == s2_lower[i])
    return matches / min_len


def get_first_significant_word(text: str) -> str:
    """
    Retourne le premier mot significatif d'une chaîne, en ignorant les articles.
    
    Args:
        text: Texte à analyser
    
    Returns:
        Premier mot significatif (sans articles comme "the", "a", "an")
    """
    words = text.lower().strip().split()
    if not words:
        return ""
    
    # Articles à ignorer
    articles = {"the", "a", "an", "le", "la", "les", "un", "une"}
    
    # Trouver le premier mot qui n'est pas un article
    for word in words:
        # Enlever la ponctuation
        word_clean = re.sub(r'[.,;:!?]+$', '', word)
        if word_clean and word_clean not in articles:
            return word_clean
    
    # Si tous les mots sont des articles, retourner le premier
    return words[0] if words else ""


def first_significant_word_similarity(s1: str, s2: str) -> Tuple[bool, float]:
    """
    Vérifie si les premiers mots significatifs (sans articles) de deux chaînes sont similaires.
    
    Returns:
        Tuple (are_similar, similarity_ratio)
        - are_similar: True si les premiers mots significatifs sont identiques ou très similaires
        - similarity_ratio: Ratio de similarité entre les premiers mots significatifs (0-1)
    """
    word1 = get_first_significant_word(s1)
    word2 = get_first_significant_word(s2)
    
    if not word1 or not word2:
        return (False, 0.0)
    
    # Si les mots sont identiques
    if word1 == word2:
        return (True, 1.0)
    
    # Calculer la similarité entre les mots
    word_similarity = string_similarity(word1, word2)
    
    # Si la similarité est très élevée (>= 0.9), considérer comme similaire
    # Cela permet de gérer les cas comme "Internation" vs "International" (typo)
    # Mais rejette "National" vs "International" (similarité ~0.5)
    if word_similarity >= 0.9:
        return (True, word_similarity)
    
    # Si les mots sont de longueur similaire et partagent un préfixe significatif
    # (ex: "Internation" vs "International" - mais pas "National" vs "International")
    if len(word1) >= 4 and len(word2) >= 4:
        prefix_len = min(len(word1), len(word2), 4)
        prefix1 = word1[:prefix_len]
        prefix2 = word2[:prefix_len]
        if prefix1 == prefix2 and word_similarity >= 0.85:
            return (True, word_similarity)
    
    return (False, word_similarity)


def are_variants(s1: str, s2: str, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """
    Détermine si deux entités sont des variantes l'une de l'autre.
    
    Critères stricts pour éviter les faux positifs:
    - "Bollywood" vs "Hollywood": rejeté car préfixes différents
    - "the National Hockey League" vs "the International Hockey League": rejeté car premiers mots significatifs différents
    - Similarité de chaîne >= threshold
    - Distance d'édition relative raisonnable
    - Les premiers mots significatifs (sans articles) doivent être similaires (>= 0.9)
    - Les préfixes doivent être similaires (au moins 2/3 des 3 premiers caractères identiques)
    - Longueur minimale respectée
    - Patterns spéciaux (initiales, abréviations)
    """
    s1_norm = normalize_entity_text(s1)
    s2_norm = normalize_entity_text(s2)
    
    # Longueur minimale
    if len(s1_norm) < MIN_LENGTH_FOR_VARIANTS or len(s2_norm) < MIN_LENGTH_FOR_VARIANTS:
        return False
    
    # Si les chaînes sont identiques (après normalisation), ce sont des variantes
    if s1_norm.lower() == s2_norm.lower():
        return True
    
    # Similarité de base
    similarity = string_similarity(s1_norm, s2_norm)
    
    # Calculer la distance d'édition
    edit_dist = edit_distance(s1_norm, s2_norm)
    max_len = max(len(s1_norm), len(s2_norm))
    edit_ratio = 1.0 - (edit_dist / max_len) if max_len > 0 else 0.0
    
    # RÈGLE 1: Si la similarité est très élevée ET la distance d'édition est faible
    # ET les préfixes sont similaires, alors probablement des variantes
    if similarity >= threshold:
        # Vérifier que les premiers mots significatifs (sans articles) sont similaires
        # Si les premiers mots significatifs diffèrent complètement, ce sont des entités différentes
        # (ex: "the National Hockey League" vs "the International Hockey League")
        first_word_sim, first_word_ratio = first_significant_word_similarity(s1_norm, s2_norm)
        if not first_word_sim:
            return False  # Premiers mots significatifs trop différents
        
        # Vérifier que les préfixes sont similaires (au moins 2-3 premiers caractères identiques)
        # Si les préfixes diffèrent beaucoup, ce sont probablement des entités différentes
        prefix_sim = prefix_similarity(s1_norm, s2_norm, min_length=3)
        
        # Si les 3 premiers caractères diffèrent, probablement pas des variantes
        # (ex: "Bollywood" vs "Hollywood" -> "Bol" vs "Hol" -> différent)
        if prefix_sim < 0.67:  # Moins de 2/3 des 3 premiers caractères identiques
            return False
        
        # Vérifier aussi que la distance d'édition relative est raisonnable
        # Pour des variantes, on s'attend à une distance d'édition relativement faible
        if edit_ratio >= 0.7:  # Au moins 70% des caractères sont identiques
            return True
    
    # RÈGLE 2: Détection de patterns d'initiales (ex: "O. Royan" vs "Olivier Royan")
    # Pattern: une lettre suivie d'un point et d'un nom
    pattern_init = r'^[A-Z]\.\s+[A-Z][a-z]+'
    if (re.match(pattern_init, s1_norm) and not re.match(pattern_init, s2_norm)) or \
       (re.match(pattern_init, s2_norm) and not re.match(pattern_init, s1_norm)):
        # Extraire le nom de famille
        parts1 = s1_norm.split()
        parts2 = s2_norm.split()
        if len(parts1) >= 2 and len(parts2) >= 2:
            # Comparer les noms de famille (dernier mot)
            if parts1[-1].lower() == parts2[-1].lower():
                # Vérifier que les préfixes du nom de famille sont identiques
                # et que la similarité globale est raisonnable
                if similarity >= 0.6 and edit_ratio >= 0.6:
                    return True
    
    # RÈGLE 3: Variantes avec différences mineures (typos, accents, etc.)
    # Mais seulement si les préfixes ET les premiers mots significatifs sont identiques
    if similarity >= 0.85 and edit_ratio >= 0.8:
        first_word_sim, _ = first_significant_word_similarity(s1_norm, s2_norm)
        if not first_word_sim:
            return False  # Premiers mots significatifs trop différents
        
        prefix_sim = prefix_similarity(s1_norm, s2_norm, min_length=2)
        if prefix_sim >= 1.0:  # Les 2 premiers caractères doivent être identiques
            return True
    
    return False


def group_variants(entities: List[Tuple[str, int]], threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, List[Tuple[str, int]]]:
    """
    Groupe les entités en clusters de variantes.
    
    Args:
        entities: Liste de tuples (texte_entité, fréquence)
        threshold: Seuil de similarité
    
    Returns:
        Dictionnaire {canonical_entity: [(variant, freq), ...]}
        où canonical_entity est la variante la plus fréquente
    """
    # Trier par fréquence décroissante (les plus fréquentes d'abord)
    entities_sorted = sorted(entities, key=lambda x: x[1], reverse=True)
    
    # Clustering simple: pour chaque entité, chercher si elle appartient à un cluster existant
    clusters = {}  # {canonical: [(variant, freq), ...]}
    
    for entity_text, freq in entities_sorted:
        entity_norm = normalize_entity_text(entity_text)
        assigned = False
        
        # Chercher dans les clusters existants
        for canonical, variants in clusters.items():
            if are_variants(entity_norm, canonical, threshold):
                # Ajouter à ce cluster
                variants.append((entity_text, freq))
                assigned = True
                break
        
        # Si pas assigné, créer un nouveau cluster
        if not assigned:
            clusters[entity_norm] = [(entity_text, freq)]
    
    # Pour chaque cluster, choisir la variante la plus fréquente comme canonical
    final_clusters = {}
    for canonical, variants in clusters.items():
        if len(variants) > 1:  # Seulement les clusters avec variantes
            # Trouver la variante la plus fréquente
            most_frequent = max(variants, key=lambda x: x[1])
            canonical_text = most_frequent[0]
            final_clusters[canonical_text] = variants
        else:
            # Pas de variantes, garder tel quel
            final_clusters[canonical] = variants
    
    return final_clusters


def analyze_variants_streaming(filepath: str, max_docs: int = None, 
                               focus_types: List[str] = None,
                               similarity_threshold: float = SIMILARITY_THRESHOLD) -> Dict:
    """
    Analyse les variantes d'entités en mode streaming.
    
    Args:
        filepath: Chemin vers le fichier JSONL
        max_docs: Nombre maximum de documents à traiter
        focus_types: Types d'entités à analyser (None = tous, recommandé: ["PERSON", "ORG", "GPE"])
        similarity_threshold: Seuil de similarité pour détecter les variantes
    
    Returns:
        Dictionnaire avec les statistiques de variantes
    """
    # Compteurs par type d'entité
    entities_by_type = defaultdict(Counter)
    
    print("Collecte des entités...")
    total_docs = 0
    
    # Première passe: collecter toutes les entités avec leurs fréquences
    for doc in tqdm(load_ner_data(filepath), desc="Lecture des documents"):
        if max_docs and total_docs >= max_docs:
            break
        
        total_docs += 1
        
        for entity in doc['entities']:
            label = entity['label']
            text = entity['text']
            
            # Filtrer par type si spécifié
            if focus_types and label not in focus_types:
                continue
            
            entities_by_type[label][text] += 1
    
    print(f"\nAnalyse des variantes pour {len(entities_by_type)} types d'entités...")
    
    # Deuxième passe: grouper les variantes par type
    variant_stats = {}
    
    for label, entity_counter in entities_by_type.items():
        print(f"\nTraitement de {label} ({len(entity_counter)} entités uniques)...")
        
        # Convertir en liste de tuples
        entities_list = list(entity_counter.items())
        
        # Grouper les variantes
        clusters = group_variants(entities_list, threshold=similarity_threshold)
        
        # Calculer les statistiques
        total_entities = sum(freq for _, freq in entities_list)
        total_unique = len(entities_list)
        clusters_with_variants = {k: v for k, v in clusters.items() if len(v) > 1}
        num_clusters_with_variants = len(clusters_with_variants)
        
        # Compter les entités affectées par les variantes
        entities_in_variants = sum(len(v) for v in clusters_with_variants.values())
        total_freq_in_variants = sum(
            sum(freq for _, freq in variants)
            for variants in clusters_with_variants.values()
        )
        
        # Top clusters avec le plus de variantes
        top_clusters = sorted(
            clusters_with_variants.items(),
            key=lambda x: (len(x[1]), sum(freq for _, freq in x[1])),
            reverse=True
        )[:20]
        
        variant_stats[label] = {
            'total_occurrences': total_entities,
            'total_unique_entities': total_unique,
            'clusters_with_variants': num_clusters_with_variants,
            'entities_affected_by_variants': entities_in_variants,
            'occurrences_affected_by_variants': total_freq_in_variants,
            'noise_ratio': round(entities_in_variants / total_unique * 100, 2) if total_unique > 0 else 0,
            'noise_ratio_by_frequency': round(total_freq_in_variants / total_entities * 100, 2) if total_entities > 0 else 0,
            'top_variant_clusters': [
                {
                    'canonical': canonical,
                    'variants': [{'text': text, 'freq': freq} for text, freq in variants],
                    'num_variants': len(variants),
                    'total_freq': sum(freq for _, freq in variants)
                }
                for canonical, variants in top_clusters
            ]
        }
    
    return {
        'total_documents_processed': total_docs,
        'similarity_threshold': similarity_threshold,
        'focus_types': focus_types if focus_types else "all",
        'variant_stats_by_type': variant_stats
    }


def print_variant_analysis(results: Dict):
    """Affiche l'analyse des variantes de manière lisible."""
    print("\n" + "="*80)
    print("ANALYSE DU BRUIT DANS LES ENTITÉS (VARIANTES)")
    print("="*80)
    print(f"Documents traités: {results['total_documents_processed']:,}")
    print(f"Seuil de similarité: {results['similarity_threshold']}")
    print(f"Types analysés: {results['focus_types']}")
    
    print("\n" + "-"*80)
    print("RÉSUMÉ PAR TYPE D'ENTITÉ")
    print("-"*80)
    
    # Trier par ratio de bruit décroissant
    sorted_types = sorted(
        results['variant_stats_by_type'].items(),
        key=lambda x: x[1]['noise_ratio'],
        reverse=True
    )
    
    for label, stats in sorted_types:
        print(f"\n{label}:")
        print(f"  Total d'occurrences: {stats['total_occurrences']:,}")
        print(f"  Entités uniques: {stats['total_unique_entities']:,}")
        print(f"  Clusters avec variantes: {stats['clusters_with_variants']:,}")
        print(f"  Entités affectées: {stats['entities_affected_by_variants']:,} "
              f"({stats['noise_ratio']:.1f}% des entités uniques)")
        print(f"  Occurrences affectées: {stats['occurrences_affected_by_variants']:,} "
              f"({stats['noise_ratio_by_frequency']:.1f}% des occurrences)")
    
    print("\n" + "-"*80)
    print("TOP 10 CLUSTERS DE VARIANTES (par nombre de variantes)")
    print("-"*80)
    
    all_top_clusters = []
    for label, stats in results['variant_stats_by_type'].items():
        for cluster in stats['top_variant_clusters']:
            all_top_clusters.append((label, cluster))
    
    # Trier par nombre de variantes
    all_top_clusters.sort(key=lambda x: x[1]['num_variants'], reverse=True)
    
    for i, (label, cluster) in enumerate(all_top_clusters[:10], 1):
        print(f"\n{i}. [{label}] {cluster['canonical']} ({cluster['num_variants']} variantes, "
              f"{cluster['total_freq']} occurrences totales):")
        for variant in cluster['variants']:
            print(f"   - \"{variant['text']}\" ({variant['freq']} occurrences)")


def save_variant_analysis(results: Dict, filepath: str):
    """Sauvegarde l'analyse des variantes dans un fichier JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyse des variantes d\'entités NER')
    parser.add_argument('--input', type=str, default='ner_results.jsonl',
                       help='Fichier d\'entrée JSONL')
    parser.add_argument('--max-docs', type=int, default=None,
                       help='Nombre maximum de documents à traiter (pour test rapide)')
    parser.add_argument('--focus-types', type=str, nargs='+', default=None,
                       help='Types d\'entités à analyser (ex: PERSON ORG GPE). Par défaut: tous')
    parser.add_argument('--similarity-threshold', type=float, default=SIMILARITY_THRESHOLD,
                       help=f'Seuil de similarité pour détecter les variantes (défaut: {SIMILARITY_THRESHOLD})')
    parser.add_argument('--output', type=str, default='results/stats/variant_analysis.json',
                       help='Fichier de sortie JSON')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Erreur: Le fichier {args.input} n'existe pas.")
        return
    
    print(f"Analyse des variantes depuis {args.input}...")
    results = analyze_variants_streaming(
        args.input,
        max_docs=args.max_docs,
        focus_types=args.focus_types,
        similarity_threshold=args.similarity_threshold
    )
    
    # Sauvegarder
    save_variant_analysis(results, args.output)
    print(f"\nRésultats sauvegardés dans {args.output}")
    
    # Afficher
    print_variant_analysis(results)

if __name__ == "__main__":
    main()

