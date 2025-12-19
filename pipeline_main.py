"""
Orchestrateur principal pour exécuter le pipeline complet.
Permet d'exécuter toutes les étapes ou des étapes spécifiques.
"""
import argparse
import os
import subprocess
import sys
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
            '--output', 'results/stats/variant_analysis.json',
            '--focus-types', 'PERSON', 'ORG', 'GPE'
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
            '--output', 'results/stats/variant_analysis_after.json',
            '--focus-types', 'PERSON', 'ORG', 'GPE'
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
    }
}


def run_step(step_num: str, extra_args: list = None, max_articles: int = None):
    """Exécute une étape spécifique du pipeline."""
    if step_num not in STEPS:
        print(f"Erreur: Étape '{step_num}' inconnue.")
        print(f"Étapes disponibles: {', '.join(STEPS.keys())}")
        return False
    
    step = STEPS[step_num]
    script = step['script']
    
    if not os.path.exists(script):
        print(f"Erreur: Le script {script} n'existe pas.")
        return False
    
    print("\n" + "="*80)
    print(f"ÉTAPE {step_num}: {step['name']}")
    print("="*80)
    
    # Utiliser les arguments par défaut de l'étape, avec possibilité de les surcharger
    cmd = [sys.executable, script] + step.get('args', [])
    
    # Ajouter --max-articles pour l'étape 1 (extraction des entités)
    if step_num == '1' and max_articles is not None:
        cmd.extend(['--max-articles', str(max_articles)])
    
    # Ajouter --max-docs pour les étapes d'analyse (2, 4, 5)
    if step_num in ['2', '4', '5'] and max_articles is not None:
        cmd.extend(['--max-docs', str(max_articles)])
    
    if extra_args:
        cmd.extend(extra_args)
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"✓ Étape {step_num} terminée avec succès")
        return True
    else:
        print(f"✗ Étape {step_num} a échoué")
        return False


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
    
    if args.all:
        steps_to_run = list(STEPS.keys())
    elif args.step:
        steps_to_run = [args.step]
    elif args.steps:
        steps_to_run = args.steps
    elif args.from_step:
        start_idx = list(STEPS.keys()).index(args.from_step)
        steps_to_run = list(STEPS.keys())[start_idx:]
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
    
    # Exécuter les étapes
    for step_num in steps_to_run:
        success = run_step(step_num, max_articles=args.max_articles)
        if not success:
            print(f"\n✗ Le pipeline s'est arrêté à l'étape {step_num}")
            return
    evaluator = Evaluator()
    print("\n" + "="*80)
    print("✓ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("="*80)
    print("Evaluating Results:")
    print("="*80)
    print("Relations.csv:")
    evaluator.evaluate_relations()
    acronyms = pd.read_csv(os.path.join("results", "csv", "acronyms.csv"), sep = ";")
    entities = pd.read_csv(os.path.join("results", "csv", "entities.csv"), sep = ";")
    evaluator.evaluate_acronyms(acronyms)
    evaluator.evaluate_entities()


if __name__ == '__main__':
    main()

