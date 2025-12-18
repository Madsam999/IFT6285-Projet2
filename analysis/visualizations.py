"""
Script pour générer les visualisations des statistiques NER.
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import os
from collections import Counter
from typing import Dict, List

# Configuration pour le français et meilleure qualité
matplotlib.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100


def load_stats(filepath: str) -> Dict:
    """Charge les statistiques depuis un fichier JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_entity_type_distribution(stats: Dict, output_dir: str = "results/figures"):
    """
    Crée un graphique en barres de la distribution des types d'entités.
    """
    entity_dist = stats['general_stats']['entity_type_distribution']
    
    labels = list(entity_dist.keys())
    counts = list(entity_dist.values())
    
    # Trier par fréquence décroissante
    sorted_data = sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_data)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(labels)), counts, color='steelblue', alpha=0.7)
    plt.xlabel('Type d\'entité', fontsize=12)
    plt.ylabel('Nombre d\'occurrences', fontsize=12)
    plt.title('Distribution des entités par type', fontsize=14, fontweight='bold')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/entity_type_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {output_dir}/entity_type_distribution.png")
    plt.close()


def plot_top_entities_by_type(stats: Dict, output_dir: str = "results/figures", top_n: int = 10):
    """
    Crée des graphiques pour les entités les plus fréquentes par type.
    """
    type_stats = stats['type_stats']
    
    # Créer un graphique pour chaque type avec au moins top_n entités
    for label, type_data in type_stats.items():
        if len(type_data['most_frequent']) < 3:
            continue
        
        entities = list(type_data['most_frequent'].items())[:top_n]
        if not entities:
            continue
        
        entity_names, counts = zip(*entities)
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(entity_names)), counts, color='coral', alpha=0.7)
        plt.ylabel('Entités', fontsize=12)
        plt.xlabel('Nombre d\'occurrences', fontsize=12)
        plt.title(f'Top {len(entities)} entités les plus fréquentes - {label}', 
                 fontsize=14, fontweight='bold')
        plt.yticks(range(len(entity_names)), entity_names)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        # Ajouter les valeurs
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}',
                    ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        safe_label = label.replace('/', '_').replace('\\', '_')
        plt.savefig(f"{output_dir}/top_entities_{safe_label}.png", dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {output_dir}/top_entities_{safe_label}.png")
        plt.close()


def plot_top_entities_all_types(stats: Dict, output_dir: str = "results/figures", top_n: int = 20):
    """
    Crée un graphique des entités les plus fréquentes (tous types confondus).
    """
    all_entities = stats['frequent_stats']['most_frequent_all_types']
    top_entities = list(all_entities.items())[:top_n]
    
    if not top_entities:
        print("Aucune entité à visualiser.")
        return
    
    entity_names, counts = zip(*top_entities)
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(entity_names)), counts, color='mediumseagreen', alpha=0.7)
    plt.ylabel('Entités', fontsize=12)
    plt.xlabel('Nombre d\'occurrences', fontsize=12)
    plt.title(f'Top {top_n} entités les plus fréquentes (tous types confondus)', 
             fontsize=14, fontweight='bold')
    plt.yticks(range(len(entity_names)), entity_names)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}',
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/top_entities_all_types.png", dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {output_dir}/top_entities_all_types.png")
    plt.close()


def plot_entities_per_document_distribution(stats: Dict, data: List[Dict], 
                                            output_dir: str = "results/figures"):
    """
    Crée un histogramme de la distribution du nombre d'entités par document.
    """
    entities_per_doc = [len(doc['entities']) for doc in data]
    
    plt.figure(figsize=(10, 6))
    plt.hist(entities_per_doc, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Nombre d\'entités par document', fontsize=12)
    plt.ylabel('Nombre de documents', fontsize=12)
    plt.title('Distribution du nombre d\'entités par document', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter des statistiques
    mean_val = sum(entities_per_doc) / len(entities_per_doc)
    median_val = sorted(entities_per_doc)[len(entities_per_doc)//2]
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {mean_val:.2f}')
    plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Médiane: {median_val:.2f}')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/entities_per_document_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {output_dir}/entities_per_document_distribution.png")
    plt.close()


def plot_variant_noise_ratio(variant_stats: Dict, output_dir: str = "results/figures"):
    """
    Crée un graphique en barres du ratio de bruit (variantes) par type d'entité.
    """
    variant_data = variant_stats['variant_stats_by_type']
    
    if not variant_data:
        print("Aucune donnée de variantes à visualiser.")
        return
    
    labels = []
    noise_ratios = []
    noise_ratios_by_freq = []
    
    # Trier par ratio de bruit décroissant
    sorted_data = sorted(
        variant_data.items(),
        key=lambda x: x[1]['noise_ratio'],
        reverse=True
    )
    
    for label, stats in sorted_data:
        labels.append(label)
        noise_ratios.append(stats['noise_ratio'])
        noise_ratios_by_freq.append(stats['noise_ratio_by_frequency'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique 1: Ratio de bruit par nombre d'entités uniques
    bars1 = ax1.bar(range(len(labels)), noise_ratios, color='crimson', alpha=0.7)
    ax1.set_xlabel('Type d\'entité', fontsize=12)
    ax1.set_ylabel('Ratio de bruit (%)', fontsize=12)
    ax1.set_title('Ratio de bruit: % d\'entités affectées par des variantes\n(par nombre d\'entités uniques)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    # Graphique 2: Ratio de bruit par fréquence d'occurrences
    bars2 = ax2.bar(range(len(labels)), noise_ratios_by_freq, color='darkorange', alpha=0.7)
    ax2.set_xlabel('Type d\'entité', fontsize=12)
    ax2.set_ylabel('Ratio de bruit (%)', fontsize=12)
    ax2.set_title('Ratio de bruit: % d\'occurrences affectées par des variantes\n(par fréquence)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/variant_noise_ratio.png", dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {output_dir}/variant_noise_ratio.png")
    plt.close()


def plot_top_variant_clusters(variant_stats: Dict, output_dir: str = "results/figures", top_n: int = 15):
    """
    Crée un graphique des clusters de variantes avec le plus de variantes.
    """
    variant_data = variant_stats['variant_stats_by_type']
    
    # Collecter tous les clusters
    all_clusters = []
    for label, stats in variant_data.items():
        for cluster in stats['top_variant_clusters']:
            all_clusters.append({
                'label': label,
                'canonical': cluster['canonical'],
                'num_variants': cluster['num_variants'],
                'total_freq': cluster['total_freq']
            })
    
    if not all_clusters:
        print("Aucun cluster de variantes à visualiser.")
        return
    
    # Trier par nombre de variantes
    all_clusters.sort(key=lambda x: x['num_variants'], reverse=True)
    top_clusters = all_clusters[:top_n]
    
    # Créer les labels pour l'affichage
    cluster_labels = [f"{c['canonical'][:30]}... ({c['label']})" if len(c['canonical']) > 30 
                     else f"{c['canonical']} ({c['label']})" 
                     for c in top_clusters]
    num_variants = [c['num_variants'] for c in top_clusters]
    total_freqs = [c['total_freq'] for c in top_clusters]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Graphique 1: Nombre de variantes par cluster
    bars1 = ax1.barh(range(len(cluster_labels)), num_variants, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Clusters de variantes', fontsize=12)
    ax1.set_xlabel('Nombre de variantes', fontsize=12)
    ax1.set_title(f'Top {top_n} clusters avec le plus de variantes', 
                 fontsize=14, fontweight='bold')
    ax1.set_yticks(range(len(cluster_labels)))
    ax1.set_yticklabels(cluster_labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}',
                ha='left', va='center', fontsize=9)
    
    # Graphique 2: Fréquence totale par cluster
    bars2 = ax2.barh(range(len(cluster_labels)), total_freqs, color='mediumseagreen', alpha=0.7)
    ax2.set_ylabel('Clusters de variantes', fontsize=12)
    ax2.set_xlabel('Fréquence totale (occurrences)', fontsize=12)
    ax2.set_title(f'Top {top_n} clusters par fréquence totale', 
                 fontsize=14, fontweight='bold')
    ax2.set_yticks(range(len(cluster_labels)))
    ax2.set_yticklabels(cluster_labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}',
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/top_variant_clusters.png", dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {output_dir}/top_variant_clusters.png")
    plt.close()


def plot_variant_cluster_size_distribution(variant_stats: Dict, output_dir: str = "results/figures"):
    """
    Crée un histogramme de la distribution de la taille des clusters de variantes.
    """
    variant_data = variant_stats['variant_stats_by_type']
    
    cluster_sizes = []
    for label, stats in variant_data.items():
        for cluster in stats['top_variant_clusters']:
            cluster_sizes.append(cluster['num_variants'])
    
    if not cluster_sizes:
        print("Aucune donnée de clusters à visualiser.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(cluster_sizes, bins=min(20, max(cluster_sizes)), color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel('Nombre de variantes par cluster', fontsize=12)
    plt.ylabel('Nombre de clusters', fontsize=12)
    plt.title('Distribution du nombre de variantes par cluster', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter des statistiques
    if cluster_sizes:
        mean_val = sum(cluster_sizes) / len(cluster_sizes)
        median_val = sorted(cluster_sizes)[len(cluster_sizes)//2]
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {mean_val:.2f}')
        plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Médiane: {median_val:.2f}')
        plt.legend()
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/distribution_nombre_variantes_par_cluster.png", dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {output_dir}/distribution_nombre_variantes_par_cluster.png")
    plt.close()


def plot_variant_analysis(variant_stats_file: str = "results/stats/variant_analysis.json",
                         output_dir: str = "results/figures"):
    """
    Génère toutes les visualisations pour l'analyse des variantes.
    """
    if not os.path.exists(variant_stats_file):
        print(f"Erreur: Le fichier {variant_stats_file} n'existe pas.")
        print("Veuillez d'abord exécuter entity_variants.py")
        return
    
    print("Chargement des statistiques de variantes...")
    variant_stats = load_stats(variant_stats_file)
    
    print("\nGénération des visualisations de variantes...")
    
    # Graphique 1: Ratio de bruit
    print("1. Ratio de bruit par type d'entité...")
    plot_variant_noise_ratio(variant_stats, output_dir)
    
    # Graphique 2: Top clusters de variantes
    print("2. Top clusters de variantes...")
    plot_top_variant_clusters(variant_stats, output_dir)
    
    # Graphique 3: Distribution de la taille des clusters
    print("3. Distribution de la taille des clusters...")
    plot_variant_cluster_size_distribution(variant_stats, output_dir)
    
    print("\nVisualisations de variantes générées avec succès!")


def main():
    """Fonction principale pour générer toutes les visualisations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Génération de visualisations NER')
    parser.add_argument('--variants', action='store_true',
                       help='Générer les visualisations des variantes d\'entités')
    parser.add_argument('--variant-file', type=str, default='results/stats/variant_analysis.json',
                       help='Fichier JSON des statistiques de variantes')
    parser.add_argument('--output-dir', type=str, default='results/figures',
                       help='Répertoire de sortie pour les graphiques')
    
    args = parser.parse_args()
    
    if args.variants:
        # Visualisations des variantes
        plot_variant_analysis(args.variant_file, args.output_dir)
    else:
        # Visualisations descriptives standard
        stats_file = "results/stats/descriptive_stats.json"
        
        if not os.path.exists(stats_file):
            print(f"Erreur: Le fichier {stats_file} n'existe pas.")
            print("Veuillez d'abord exécuter descriptive_stats.py")
            return
        
        print("Chargement des statistiques...")
        stats = load_stats(stats_file)
        
        print("\nGénération des visualisations...")
        
        # Graphique 1: Distribution par type
        print("1. Distribution des types d'entités...")
        plot_entity_type_distribution(stats, args.output_dir)
        
        # Graphique 2: Top entités par type
        print("2. Top entités par type...")
        plot_top_entities_by_type(stats, args.output_dir)
        
        # Graphique 3: Top entités tous types confondus
        print("3. Top entités (tous types)...")
        plot_top_entities_all_types(stats, args.output_dir)
        
        print("\nVisualisations générées avec succès!")


if __name__ == "__main__":
    main()

