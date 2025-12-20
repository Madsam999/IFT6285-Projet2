"""
Orchestrateur principal pour exécuter le pipeline complet.
Permet d'exécuter toutes les étapes ou des étapes spécifiques.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import pandas as pd
from evaluation.information_evaluator import Evaluator


STEPS = {
    '1': {
        'name': 'Extraction des entités',
        'script': 'extraction/extract_entities.py',
        'output': 'results/raw/entities_raw.jsonl',
        'args': ['--output', 'results/raw/entities_raw.jsonl']
    },
    '2': {
        'name': 'Analyse du bruit (avant normalisation)',
        'script': 'analysis/entity_variants.py',
        'output': 'results/stats/variant_analysis.json',
        'args': [
            '--input', 'results/raw/entities_raw.jsonl',
            '--output', 'results/stats/variant_analysis.json'
        ]
    },
    '3': {
        'name': 'Réduction du bruit',
        'script': 'normalization/reduce_noise.py',
        'output': 'results/normalized/entities_normalized.jsonl',
        'args': [
            '--input', 'results/raw/entities_raw.jsonl',
            '--output', 'results/normalized/entities_normalized.jsonl',
            '--variant-analysis', 'results/stats/variant_analysis.json'
        ]
    },
    '4': {
        'name': 'Analyse du bruit (après normalisation)',
        'script': 'analysis/entity_variants.py',
        'output': 'results/stats/variant_analysis_after.json',
        'args': [
            '--input', 'results/normalized/entities_normalized.jsonl',
            '--output', 'results/stats/variant_analysis_after.json'
        ]
    },
    '4b': {
        'name': 'Statistiques descriptives',
        'script': 'analysis/descriptive_stats.py',
        'output': 'results/stats/descriptive_stats.json',
        'args': [
            '--input', 'results/normalized/entities_normalized.jsonl',
            '--no-cooccurrences'
        ]
    },
    '5': {
        'name': 'Extraction des relations (Stanza)',
        'script': 'extraction/extract_relations_stanza.py',
        'output': 'results/raw/relations.jsonl',
        'args': [
            '--input', 'results/normalized/entities_normalized.jsonl',
            '--output', 'results/raw/relations.jsonl'
        ]
    },
    '6': {
        'name': 'Export vers CSV',
        'script': 'export/export_to_csv.py',
        'output': 'results/csv/',
        'args': [
            '--entities-input', 'results/normalized/entities_normalized.jsonl',
            '--relations-input', 'results/raw/relations.jsonl',
            '--output-dir', 'results/csv'
        ]
    },
    '7': {
        'name': 'Génération des visualisations',
        'script': 'analysis/generate_all_plots.py',
        'output': 'results/figures/',
        'args': []
    }
}


def format_time(seconds: float) -> str:
    """Formate le temps en format lisible (heures:minutes:secondes ou minutes:secondes)."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def run_step(step_num: str, extra_args: list = None, max_articles: int = None):
    """
    Exécute une étape spécifique du pipeline.
    
    Returns:
        tuple: (success: bool, runtime: float) - Succès de l'étape et temps d'exécution en secondes
    """
    if step_num not in STEPS:
        print(f"Erreur: Étape '{step_num}' inconnue.")
        print(f"Étapes disponibles: {', '.join(STEPS.keys())}")
        return False, 0.0
    
    step = STEPS[step_num]
    script = step['script']
    
    if not os.path.exists(script):
        print(f"Erreur: Le script {script} n'existe pas.")
        return False, 0.0
    
    print("\n" + "="*80)
    print(f"ÉTAPE {step_num}: {step['name']}")
    print("="*80)
    
    # Utiliser les arguments par défaut de l'étape, avec possibilité de les surcharger
    cmd = [sys.executable, script] + step.get('args', [])
    
    # Ajouter --max-articles pour l'étape 1 (extraction des entités)
    if step_num == '1' and max_articles is not None:
        cmd.extend(['--max-articles', str(max_articles)])
    
    # Ajouter --max-docs pour les étapes d'analyse (2, 4, 4b, 5)
    if step_num in ['2', '4', '4b', '5'] and max_articles is not None:
        cmd.extend(['--max-docs', str(max_articles)])
    
    if extra_args:
        cmd.extend(extra_args)
    
    # Mesurer le temps d'exécution
    start_time = time.time()
    result = subprocess.run(cmd)
    end_time = time.time()
    runtime = end_time - start_time
    
    if result.returncode == 0:
        print(f"✓ Étape {step_num} terminée avec succès")
        print(f"⏱  Temps d'exécution: {format_time(runtime)}")
        return True, runtime
    else:
        print(f"✗ Étape {step_num} a échoué")
        print(f"⏱  Temps d'exécution: {format_time(runtime)}")
        return False, runtime


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline complet d\'extraction d\'information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Exécuter toutes les étapes')
    parser.add_argument('--step', type=str,
                       help='Exécuter une étape spécifique (1-6)')
    parser.add_argument('--steps', nargs='+',
                       help='Exécuter plusieurs étapes (ex: --steps 1 2 3)')
    parser.add_argument('--from-step', type=str,
                       help='Exécuter depuis une étape jusqu\'à la fin')
    parser.add_argument('--max-articles', type=int, default=None,
                       help='Nombre maximum d\'articles à traiter (pour prototypage rapide). '
                            'S\'applique à l\'étape 1 (extraction) et aux étapes d\'analyse (2, 4)')
    parser.add_argument('--list', action='store_true',
                       help='Lister toutes les étapes disponibles')
    
    args = parser.parse_args()
    
    if args.list:
        print("Étapes disponibles:")
        for num, step in STEPS.items():
            print(f"  {num}. {step['name']}")
            print(f"     Script: {step['script']}")
            print(f"     Output: {step['output']}")
            print()
        return
    
    # Ordre des étapes (pour gérer 4b correctement)
    step_order = ['1', '2', '3', '4', '4b', '5', '6', '7']
    
    if args.all:
        steps_to_run = [s for s in step_order if s in STEPS]
    elif args.step:
        steps_to_run = [args.step]
    elif args.steps:
        steps_to_run = args.steps
    elif args.from_step:
        if args.from_step in step_order:
            start_idx = step_order.index(args.from_step)
            steps_to_run = [s for s in step_order[start_idx:] if s in STEPS]
        else:
            steps_to_run = [args.from_step]
    else:
        parser.print_help()
        return
    
    print("="*80)
    print("PIPELINE D'EXTRACTION D'INFORMATION")
    print("="*80)
    print(f"Étapes à exécuter: {', '.join(steps_to_run)}")
    if args.max_articles:
        print(f"Nombre maximum d'articles: {args.max_articles:,}")
    print()
    
    # Suivi des temps d'exécution
    step_times = {}
    pipeline_start_time = time.time()
    
    # Exécuter les étapes
    for step_num in steps_to_run:
        success, runtime = run_step(step_num, max_articles=args.max_articles)
        step_times[step_num] = runtime
        if not success:
            print(f"\n✗ Le pipeline s'est arrêté à l'étape {step_num}")
            pipeline_end_time = time.time()
            total_time = pipeline_end_time - pipeline_start_time
            print(f"\n⏱  Temps total écoulé: {format_time(total_time)}")
            return
    
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    
    # Afficher le résumé des temps
    print("\n" + "="*80)
    print("RÉSUMÉ DES TEMPS D'EXÉCUTION")
    print("="*80)
    for step_num in steps_to_run:
        step_name = STEPS[step_num]['name']
        runtime = step_times[step_num]
        print(f"Étape {step_num}: {step_name:50s} {format_time(runtime):>15s}")
    print("-"*80)
    print(f"{'TOTAL':50s} {format_time(total_time):>15s}")
    print("="*80)
    
    # Collecter les statistiques pour le rapport
    print("\n" + "="*80)
    print("RÉSUMÉ DES STATISTIQUES")
    print("="*80)
    
    summary_stats = {
        'pipeline_times': {step_num: step_times[step_num] for step_num in steps_to_run},
        'total_time': total_time,
        'max_articles': args.max_articles
    }
    
    # Extraire les ratios de bruit avant et après normalisation
    import json as json_module  # Import explicite pour éviter les conflits de scope
    variant_before_file = 'results/stats/variant_analysis.json'
    variant_after_file = 'results/stats/variant_analysis_after.json'
    
    noise_ratios_before = {}
    noise_ratios_after = {}
    
    if os.path.exists(variant_before_file):
        try:
            with open(variant_before_file, 'r', encoding='utf-8') as f:
                variant_data_before = json_module.load(f)
                variant_stats_before = variant_data_before.get('variant_stats_by_type', {})
                for entity_type, stats in variant_stats_before.items():
                    noise_ratios_before[entity_type] = {
                        'by_entities': stats.get('noise_ratio', 0),
                        'by_frequency': stats.get('noise_ratio_by_frequency', 0)
                    }
        except Exception as e:
            print(f"Impossible de lire les ratios de bruit (avant): {e}")
    
    if os.path.exists(variant_after_file):
        try:
            with open(variant_after_file, 'r', encoding='utf-8') as f:
                variant_data_after = json_module.load(f)
                variant_stats_after = variant_data_after.get('variant_stats_by_type', {})
                for entity_type, stats in variant_stats_after.items():
                    noise_ratios_after[entity_type] = {
                        'by_entities': stats.get('noise_ratio', 0),
                        'by_frequency': stats.get('noise_ratio_by_frequency', 0)
                    }
        except Exception as e:
            print(f"Impossible de lire les ratios de bruit (après): {e}")
    
    # Afficher les ratios de bruit
    if noise_ratios_before or noise_ratios_after:
        print("\nRatios de bruit (variantes d'entités):")
        print("-" * 80)
        
        # Collecter tous les types d'entités trouvés
        all_types = set()
        if noise_ratios_before:
            all_types.update(noise_ratios_before.keys())
        if noise_ratios_after:
            all_types.update(noise_ratios_after.keys())
        
        # Trier les types pour un affichage cohérent
        sorted_types = sorted(all_types)
        
        if noise_ratios_before:
            print("\nAVANT normalisation:")
            for entity_type in sorted_types:
                if entity_type in noise_ratios_before:
                    ratio = noise_ratios_before[entity_type]['by_entities']
                    print(f"  {entity_type:15s}: {ratio:5.2f}%")
        
        if noise_ratios_after:
            print("\nAPRÈS normalisation:")
            for entity_type in sorted_types:
                if entity_type in noise_ratios_after:
                    ratio = noise_ratios_after[entity_type]['by_entities']
                    print(f"  {entity_type:15s}: {ratio:5.2f}%")
        
        # Calculer la réduction
        if noise_ratios_before and noise_ratios_after:
            print("\nRÉDUCTION du bruit:")
            for entity_type in sorted_types:
                if entity_type in noise_ratios_before and entity_type in noise_ratios_after:
                    before = noise_ratios_before[entity_type]['by_entities']
                    after = noise_ratios_after[entity_type]['by_entities']
                    reduction = before - after
                    
                    print(f"  {entity_type:15s}: -{reduction:5.2f}%")
        
        summary_stats['noise_ratios_before'] = noise_ratios_before
        summary_stats['noise_ratios_after'] = noise_ratios_after
    
    # Compter les chunks si le fichier existe
    entities_file = 'results/raw/entities_raw.jsonl'
    if os.path.exists(entities_file):
        try:
            from utils.count_stats import count_chunks_fast, estimate_sentences_from_chunks
            num_chunks = count_chunks_fast(entities_file)
            estimated_sentences = estimate_sentences_from_chunks(num_chunks)
            summary_stats['num_chunks'] = num_chunks
            summary_stats['estimated_sentences'] = estimated_sentences
            print(f"Nombre de chunks (documents): {num_chunks:,}")
            print(f"Estimation du nombre de phrases: ~{estimated_sentences:,}")
        except Exception as e:
            print(f"Impossible de compter les chunks: {e}")
    
    # Compter les entités si le fichier CSV existe
    entities_csv = 'results/csv/entities.csv'
    if os.path.exists(entities_csv):
        try:
            entities_df = pd.read_csv(entities_csv, sep=';')
            total_entities = len(entities_df)
            summary_stats['total_entities'] = total_entities
            print(f"Nombre total d'entités extraites: {total_entities:,}")
        except Exception as e:
            print(f"Impossible de compter les entités: {e}")
    
    # Compter les relations si le fichier CSV existe
    relations_csv = 'results/csv/relations.csv'
    if os.path.exists(relations_csv):
        try:
            relations_df = pd.read_csv(relations_csv, sep=';')
            total_relations = len(relations_df)
            summary_stats['total_relations'] = total_relations
            if 'relation_type' in relations_df.columns:
                relation_types = relations_df['relation_type'].value_counts().to_dict()
                summary_stats['relation_types'] = relation_types
                print(f"Nombre total de relations extraites: {total_relations:,}")
                print("Distribution par type:")
                for rel_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {rel_type}: {count:,}")
        except Exception as e:
            print(f"Impossible de compter les relations: {e}")
    
    # Compter les acronymes si le fichier CSV existe
    acronyms_csv = 'results/csv/acronyms.csv'
    if os.path.exists(acronyms_csv):
        try:
            acronyms_df = pd.read_csv(acronyms_csv, sep=';')
            total_acronyms = len(acronyms_df)
            # Le fichier utilise 'full_form' au lieu de 'definition'
            if 'full_form' in acronyms_df.columns:
                acronyms_with_def = len(acronyms_df[
                    (acronyms_df['full_form'].notna()) & 
                    (acronyms_df['full_form'] != '') & 
                    (acronyms_df['full_form'].str.lower() != 'unknown')
                ])
            else:
                acronyms_with_def = 0
            summary_stats['total_acronyms'] = total_acronyms
            summary_stats['acronyms_with_definition'] = acronyms_with_def
            print(f"Nombre total d'acronymes extraits: {total_acronyms:,}")
            if total_acronyms > 0:
                print(f"Acronymes avec définition: {acronyms_with_def:,} ({100*acronyms_with_def/total_acronyms:.1f}%)")
        except Exception as e:
            print(f"Impossible de compter les acronymes: {e}")
    
    # Sauvegarder le résumé dans un fichier JSON pour référence
    summary_file = 'results/stats/pipeline_summary.json'
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print(f"\nRésumé sauvegardé dans: {summary_file}")
    print("="*80)
    
    evaluator = Evaluator()
    print("\n" + "="*80)
    print("✓ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("="*80)
    print("Évaluation de la qualité:")
    print("="*80)
    
    # Évaluer les relations
    print("\nRelations.csv:")
    relations_stats = evaluator.evaluate_relations()
    
    # Évaluer les acronymes
    acronyms = pd.read_csv(os.path.join("results", "csv", "acronyms.csv"), sep = ";")
    acronyms_stats = evaluator.evaluate_acronyms(acronyms)
    
    # Évaluer les entités
    entities = pd.read_csv(os.path.join("results", "csv", "entities.csv"), sep = ";")
    entities_stats = evaluator.evaluate_entities()
    
    # Sauvegarder les résultats d'évaluation
    evaluation_file = 'results/stats/evaluation_results.json'
    os.makedirs(os.path.dirname(evaluation_file), exist_ok=True)
    evaluation_results = {
        'relations': relations_stats if relations_stats else {},
        'acronyms': acronyms_stats if acronyms_stats else {},
        'entities': entities_stats if entities_stats else {}
    }
    with open(evaluation_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Résultats d'évaluation sauvegardés dans: {evaluation_file}")


if __name__ == '__main__':
    main()

