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


def main():
    """Fonction principale pour générer toutes les visualisations."""
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
    plot_entity_type_distribution(stats)
    
    # Graphique 2: Top entités par type
    print("2. Top entités par type...")
    plot_top_entities_by_type(stats)
    
    # Graphique 3: Top entités tous types confondus
    print("3. Top entités (tous types)...")
    plot_top_entities_all_types(stats)
    
    print("\nVisualisations générées avec succès!")


if __name__ == "__main__":
    main()

