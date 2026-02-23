"""
Experimental framework for testing hallucination theory with real LLMs.

Implements the core experiments from the paper:
1. IIV Classification: Test binary classification on valid vs invalid facts
2. Generative Error: Measure hallucination rate on generated responses
3. Evaluation Metric Impact: Test how scoring schemes affect behavior
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import statistics

from .data import BirthdayDataset, DatasetConfig
from .ollama_client import OllamaClient
from .evaluation import ScoringScheme, MetricsCalculator, evaluate_response


@dataclass
class ExperimentConfig:
    """Configuration for experiment."""
    model: str = "llama2"
    ollama_host: str = "http://localhost:11434"
    dataset_config: Optional[DatasetConfig] = None
    n_test_samples: int = 20
    scoring_scheme: ScoringScheme = ScoringScheme.BINARY
    penalty: float = 1.0


class HallucinationExperiment:
    """Run experiments to validate hallucination theory."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset = BirthdayDataset(config.dataset_config or DatasetConfig())
        self.client = OllamaClient(model=config.model, host=config.ollama_host)
        self.metrics = MetricsCalculator()
        self.results: List[Dict[str, Any]] = []
    
    def run_generation_experiment(self, n_samples: Optional[int] = None) -> None:
        """
        Test generative error: Ask model to generate birthdays.
        
        Theory: Model should hallucinate on singletons.
        """
        n_samples = n_samples or self.config.n_test_samples
        
        print(f"\n{'='*60}")
        print(f"GENERATIVE ERROR EXPERIMENT")
        print(f"{'='*60}")
        print(f"Model: {self.config.model}")
        print(f"Samples: {n_samples}")
        print(f"Scoring: {self.config.scoring_scheme.value}")
        
        # Sample people: mix of singletons, unseen, and memorized
        singletons_sample = self.dataset.singletons[:n_samples//3] if len(self.dataset.singletons) >= n_samples//3 else self.dataset.singletons
        unseen_sample = self.dataset.unseen[:n_samples//3] if len(self.dataset.unseen) >= n_samples//3 else self.dataset.unseen
        memorized = [p for p in range(self.dataset.config.n_people) if self.dataset.is_memorized(p)]
        memorized_sample = memorized[:n_samples//3] if len(memorized) >= n_samples//3 else memorized
        
        test_people = singletons_sample + unseen_sample + memorized_sample
        
        print(f"\nSampling:")
        print(f"  Singletons: {len(singletons_sample)}")
        print(f"  Unseen: {len(unseen_sample)}")
        print(f"  Memorized (2+): {len(memorized_sample)}")
        
        singleton_hallucinations = 0
        unseen_hallucinations = 0
        memorized_hallucinations = 0
        errors_encountered = 0
        
        print(f"\nGenerating responses...")
        for i, person_id in enumerate(test_people):
            true_date = self.dataset.get_person_birthday(person_id)
            context = f"Person {person_id} in training corpus {self.dataset.get_mention_count(person_id)} times"
            
            # Generate birthday
            response_str = self.client.generate_birthday(f"Person_{person_id}", context)
            
            if response_str is None or len(response_str) == 0:
                errors_encountered += 1
                continue
            
            # For this demo, parse as a simple date value (0-364)
            # In real scenario, would need to parse "March 15" -> date value
            try:
                # Simple parsing: assume response contains a number
                import re
                numbers = re.findall(r'\d+', response_str)
                if numbers:
                    response_date = int(numbers[0]) % 365
                else:
                    response_date = -1  # IDK
            except:
                response_date = -1
            
            # Evaluate
            evaluation = evaluate_response(
                response_date,
                true_date,
                self.config.scoring_scheme,
                self.config.penalty
            )
            self.metrics.add_evaluation(evaluation)
            
            # Track by category
            if self.dataset.is_singleton(person_id) and evaluation.is_hallucination:
                singleton_hallucinations += 1
            elif self.dataset.is_unseen(person_id) and evaluation.is_hallucination:
                unseen_hallucinations += 1
            elif self.dataset.is_memorized(person_id) and evaluation.is_hallucination:
                memorized_hallucinations += 1
            
            result = {
                "person_id": person_id,
                "true_date": true_date,
                "response": response_str,
                "response_date": response_date,
                "evaluation": evaluation.label,
                "mention_count": self.dataset.get_mention_count(person_id),
                "is_singleton": self.dataset.is_singleton(person_id),
            }
            self.results.append(result)
            
            if (i + 1) % max(1, n_samples // 5) == 0:
                print(f"  Progress: {i+1}/{len(test_people)}")
        
        print(f"\nResults:")
        print(f"  Errors: {errors_encountered}")
        print(f"  Singleton hallucination rate: {singleton_hallucinations}/{len(singletons_sample)}")
        print(f"  Unseen hallucination rate: {unseen_hallucinations}/{len(unseen_sample)}")
        print(f"  Memorized hallucination rate: {memorized_hallucinations}/{len(memorized_sample)}")
        self.metrics.print_metrics()
    
    def run_iiv_classification_experiment(self, n_samples: Optional[int] = None) -> None:
        """
        Test IIV (Is-It-Valid) binary classification.
        
        Theory: Hard classification tasks produce high hallucination rate.
        """
        n_samples = n_samples or self.config.n_test_samples
        
        print(f"\n{'='*60}")
        print(f"IIV CLASSIFICATION EXPERIMENT")
        print(f"{'='*60}")
        print(f"Model: {self.config.model}")
        print(f"Samples: {n_samples}")
        
        # Generate IIV dataset
        iiv_data = self.dataset.generate_iiv_dataset(n_samples)
        
        correct_classifications = 0
        print(f"\nClassifying {len(iiv_data)} examples...")
        
        for i, (person_id, date, is_valid) in enumerate(iiv_data):
            person_name = f"Person_{person_id}"
            
            # Ask model to classify
            classification, confidence = self.client.classify_birthday(person_name, str(date))
            
            is_correct_pred = (classification == "valid") == is_valid
            if is_correct_pred:
                correct_classifications += 1
            
            result = {
                "person_id": person_id,
                "date": date,
                "true_label": is_valid,
                "prediction": classification,
                "confidence": confidence,
                "correct": is_correct_pred,
            }
            self.results.append(result)
            
            if (i + 1) % max(1, len(iiv_data) // 5) == 0:
                print(f"  Progress: {i+1}/{len(iiv_data)}")
        
        iiv_error_rate = 1.0 - (correct_classifications / len(iiv_data))
        
        print(f"\nIIV Classification Results:")
        print(f"  Accuracy: {correct_classifications}/{len(iiv_data)} = {correct_classifications/len(iiv_data):.1%}")
        print(f"  Error rate: {iiv_error_rate:.1%}")
        print(f"\nTheory prediction:")
        print(f"  Generative error ≥ 2 × IIV error = 2 × {iiv_error_rate:.3f} = {2*iiv_error_rate:.3f}")
    
    def compare_evaluation_metrics(
        self,
        n_samples: Optional[int] = None
    ) -> None:
        """
        Compare binary vs penalized scoring.
        
        Theory: Penalized scoring encourages "IDK" responses,
        while binary scoring incentivizes guessing.
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION METRIC COMPARISON")
        print(f"{'='*60}")
        print(f"Model: {self.config.model}")
        
        n_test = n_samples or self.config.n_test_samples
        
        # Would run same test with different scoring schemes
        print(f"Binary scheme: Rewards correct, zeros everything else")
        print(f"Penalized scheme: Rewards correct, penalizes wrong, zeros IDK")
        print(f"\nTheory: Binary scheme should see more hallucinations")
        print(f"        because model is incentivized to guess")


class ExperimentRunner:
    """Orchestrate multiple experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def run_all(self) -> None:
        """Run all experiments."""
        print("\n" + "="*60)
        print("HALLUCINATION THEORY VALIDATION")
        print("="*60)
        print(f"Paper: Kalai, Nachum, Vempala, Zhang (2025)")
        print(f"Model: {self.config.model}")
        
        # Check connection
        client = OllamaClient(model=self.config.model, host=self.config.ollama_host)
        print(f"\nChecking Ollama connection...")
        if not client.test_connection():
            print(f"ERROR: Cannot connect to Ollama at {self.config.ollama_host}")
            print(f"Make sure Ollama is running and {self.config.model} is pulled.")
            return
        
        print(f"✓ Connected to Ollama, model {self.config.model} available")
        
        # Run experiments
        exp = HallucinationExperiment(self.config)
        
        try:
            exp.run_generation_experiment()
            exp.run_iiv_classification_experiment()
            exp.compare_evaluation_metrics()
        except Exception as e:
            print(f"\nError during experiments: {e}")
            import traceback
            traceback.print_exc()
