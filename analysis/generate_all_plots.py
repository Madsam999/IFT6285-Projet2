"""
Script pour générer toutes les visualisations nécessaires pour le rapport.
Génère à la fois les visualisations descriptives et les visualisations de variantes.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.visualizations import (
    load_stats, plot_entity_type_distribution, plot_top_entities_by_type,
    plot_top_entities_all_types, plot_variant_analysis
)


def main():
    output_dir = "results/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("GÉNÉRATION DE TOUTES LES VISUALISATIONS")
    print("="*80)
    
    # 1. Visualisations descriptives
    stats_file = "results/stats/descriptive_stats.json"
    if os.path.exists(stats_file):
        print("\n1. Visualisations descriptives...")
        print("-" * 80)
        stats = load_stats(stats_file)
        
        print("   - Distribution des types d'entités...")
        plot_entity_type_distribution(stats, output_dir)
        
        print("   - Top entités par type...")
        plot_top_entities_by_type(stats, output_dir)
        
        print("   - Top entités (tous types confondus)...")
        plot_top_entities_all_types(stats, output_dir)
        
        print("   ✓ Visualisations descriptives générées")
    else:
        print(f"\n⚠ Fichier {stats_file} introuvable, visualisations descriptives ignorées")
    
    # 2. Visualisations de variantes (après normalisation - principal)
    variant_after_file = "results/stats/variant_analysis_after.json"
    if os.path.exists(variant_after_file):
        print("\n2. Visualisations de variantes (après normalisation)...")
        print("-" * 80)
        plot_variant_analysis(variant_after_file, output_dir)
        print("   ✓ Visualisations de variantes (après) générées")
    else:
        print(f"\n⚠ Fichier {variant_after_file} introuvable, visualisations de variantes ignorées")
    
    print("\n" + "="*80)
    print("✓ TOUTES LES VISUALISATIONS GÉNÉRÉES")
    print("="*80)
    print(f"Graphiques sauvegardés dans: {output_dir}")


if __name__ == "__main__":
    main()

