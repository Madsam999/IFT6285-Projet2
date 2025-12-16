"""
Utilitaire pour charger les données NER depuis le fichier JSONL
"""
import json
from typing import Iterator, Dict, List, Any
from tqdm import tqdm


def load_ner_data(filepath: str) -> Iterator[Dict[str, Any]]:
    """
    Charge les données NER depuis un fichier JSONL.
    
    Args:
        filepath: Chemin vers le fichier JSONL
        
    Yields:
        Dictionnaire avec les clés 'text' et 'entities'
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_ner_data_with_progress(filepath: str, desc: str = "Chargement des données") -> List[Dict[str, Any]]:
    """
    Charge toutes les données NER depuis un fichier JSONL avec barre de progression.
    
    Args:
        filepath: Chemin vers le fichier JSONL
        desc: Description pour la barre de progression
        
    Returns:
        Liste de dictionnaires avec les clés 'text' et 'entities'
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        # Compter les lignes d'abord pour la barre de progression
        total_lines = sum(1 for _ in f)
        f.seek(0)
        
        for line in tqdm(f, total=total_lines, desc=desc):
            if line.strip():
                data.append(json.loads(line))
    
    return data


def count_lines(filepath: str) -> int:
    """
    Compte le nombre de lignes dans un fichier JSONL.
    
    Args:
        filepath: Chemin vers le fichier JSONL
        
    Returns:
        Nombre de lignes
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

