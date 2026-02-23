"""
Dataset generation for birthday hallucination experiments.

Implements the synthetic dataset from the paper:
- N people with random birthdays
- Training corpus with varying mention frequencies
- Singleton facts (mentioned once) are prone to hallucination
"""

import random
from dataclasses import dataclass
from typing import List, Tuple
import collections


@dataclass
class DatasetConfig:
    """Configuration for synthetic birthday dataset."""
    
    n_people: int = 200
    n_dates: int = 365
    n_docs: int = 400
    seed: int = 42


class BirthdayDataset:
    """
    Synthetic dataset of people and birthdays.
    
    Follows the paper's setup:
    - Each person has a random birthday in {0..n_dates-1}
    - Training corpus has n_docs documents mentioning various people
    - People have exponential fame distribution (Zipfian)
    - Singletons (mentioned once) cannot be distinguished from wrong dates
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        random.seed(config.seed)
        
        # Generate fame distribution (exponential)
        self.fame = [random.expovariate(1.0) for _ in range(config.n_people)]
        fame_total = sum(self.fame)
        self.fame_probs = [f / fame_total for f in self.fame]
        
        # Assign random birthdays
        self.true_birthdays = [
            random.randint(0, config.n_dates - 1) 
            for _ in range(config.n_people)
        ]
        
        # Generate training corpus
        self.training_corpus: List[Tuple[int, int]] = []
        for _ in range(config.n_docs):
            person = random.choices(range(config.n_people), weights=self.fame_probs)[0]
            self.training_corpus.append((person, self.true_birthdays[person]))
        
        # Analyze mention patterns
        self.mention_count = collections.Counter(p for p, _ in self.training_corpus)
        
        # Identify singletons and unseen
        self.singletons = [
            p for p in range(config.n_people) 
            if self.mention_count[p] == 1
        ]
        self.unseen = [
            p for p in range(config.n_people) 
            if self.mention_count[p] == 0
        ]
        self.singleton_rate = len(self.singletons) / config.n_docs
        
    def get_person_birthday(self, person_id: int) -> int:
        """Get the true birthday for a person."""
        return self.true_birthdays[person_id]
    
    def get_mention_count(self, person_id: int) -> int:
        """Get how many times this person appears in training corpus."""
        return self.mention_count[person_id]
    
    def is_singleton(self, person_id: int) -> bool:
        """Check if person was mentioned exactly once."""
        return person_id in self.singletons
    
    def is_unseen(self, person_id: int) -> bool:
        """Check if person was never mentioned."""
        return person_id in self.unseen
    
    def is_memorized(self, person_id: int, threshold: int = 2) -> bool:
        """Check if person was mentioned >= threshold times (likely memorized)."""
        return self.mention_count[person_id] >= threshold
    
    def generate_iiv_dataset(self, n_examples: int = 100) -> List[Tuple[int, int, bool]]:
        """
        Generate Is-It-Valid (IIV) binary classification dataset.
        
        50% valid (correct birthday), 50% errors (random dates).
        
        Returns:
            List of (person_id, date, is_valid) tuples
        """
        iiv_data = []
        
        # Sample people uniformly
        sampled_people = random.choices(range(self.config.n_people), k=n_examples // 2)
        
        # Add valid examples
        for person in sampled_people:
            iiv_data.append((person, self.get_person_birthday(person), True))
        
        # Add error examples (random dates)
        error_people = random.choices(range(self.config.n_people), k=n_examples // 2)
        for person in error_people:
            # Generate a wrong date
            correct = self.get_person_birthday(person)
            wrong_date = random.randint(0, self.config.n_dates - 1)
            while wrong_date == correct:
                wrong_date = random.randint(0, self.config.n_dates - 1)
            iiv_data.append((person, wrong_date, False))
        
        random.shuffle(iiv_data)
        return iiv_data
    
    def __repr__(self) -> str:
        return (
            f"BirthdayDataset(n_people={self.config.n_people}, "
            f"n_docs={self.config.n_docs}, "
            f"singletons={len(self.singletons)}, "
            f"unseen={len(self.unseen)})"
        )
    
    def print_stats(self) -> None:
        """Print dataset statistics."""
        print(f"Dataset Stats:")
        print(f"  Total people: {self.config.n_people}")
        print(f"  Total docs: {self.config.n_docs}")
        print(f"  Possible dates: {self.config.n_dates}")
        print(f"  Singletons (mentioned 1x): {len(self.singletons)}")
        print(f"  Unseen (never mentioned): {len(self.unseen)}")
        print(f"  Memorized (2+ mentions): {sum(1 for p in range(self.config.n_people) if self.mention_count[p] >= 2)}")
        print(f"  Singleton rate: {self.singleton_rate:.3f}")
