"""
Ollama client wrapper for LLM interaction.

Provides high-level interface to run generation and classification tasks
on models served via Ollama.
"""

import re
from typing import Optional, Dict, Any
import ollama


class OllamaClient:
    """Interface for querying local LLMs via Ollama."""
    
    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434"):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (must be pulled in Ollama)
            host: Ollama server address
        """
        self.model = model
        self.host = host
        self.client = ollama.Client(host=host)
    
    def generate_birthday(self, person_name: str, context: str = "") -> Optional[str]:
        """
        Ask model to generate a birthday for a person.
        
        Args:
            person_name: Name of person
            context: Optional context about the person
            
        Returns:
            Generated birthday string or None if parsing fails
        """
        prompt = f"""Given this context: "{context}"

What is {person_name}'s birthday? 
Answer with ONLY a month and day (e.g., "March 15" or "12/05"), nothing else."""
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
            )
            return response.get("response", "").strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
    
    def classify_birthday(self, person_name: str, proposed_date: str) -> tuple[str, float]:
        """
        Binary classification: Is the proposed birthday valid?
        
        Args:
            person_name: Name of person
            proposed_date: Proposed birthday to classify
            
        Returns:
            (classification, confidence) where classification is "valid" or "invalid"
        """
        prompt = f"""Given that {person_name} might have birthday {proposed_date}, 
is this a plausible/valid birthday?

You are a truth classifier. Answer with ONLY "valid" or "invalid", nothing else."""
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
            )
            text = response.get("response", "").strip().lower()
            
            # Extract confidence from log probabilities if available
            confidence = 0.5
            if "valid" in text:
                return "valid", confidence
            elif "invalid" in text:
                return "invalid", confidence
            else:
                return "unknown", confidence
        except Exception as e:
            print(f"Error classifying: {e}")
            return "error", 0.0
    
    def judge_hallucination(
        self, 
        prompt: str, 
        response: str, 
        true_fact: Optional[str] = None
    ) -> tuple[str, float]:
        """
        Judge if a response is a hallucination.
        
        Args:
            prompt: The original question
            response: The model's response
            true_fact: Optional ground truth to compare against
            
        Returns:
            (judgment, confidence) where judgment is "hallucination" or "valid"
        """
        if true_fact:
            comparison = f"\nThe true fact is: {true_fact}"
        else:
            comparison = ""
        
        prompt_text = f"""Prompt: {prompt}
Response: {response}{comparison}

Is this response a hallucination (false statement presented as fact)? 
Answer with ONLY "hallucination" or "valid", nothing else."""
        
        try:
            response_text = self.client.generate(
                model=self.model,
                prompt=prompt_text,
                stream=False,
            )
            text = response_text.get("response", "").strip().lower()
            
            if "hallucination" in text:
                return "hallucination", 0.5
            elif "valid" in text:
                return "valid", 0.5
            else:
                return "unknown", 0.5
        except Exception as e:
            print(f"Error judging: {e}")
            return "error", 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            # Try to show model details via Ollama
            return {
                "model": self.model,
                "host": self.host,
                "status": "available"
            }
        except Exception as e:
            return {
                "model": self.model,
                "host": self.host,
                "status": f"unavailable: {e}"
            }
    
    def test_connection(self) -> bool:
        """Test if Ollama is running and model is available."""
        try:
            response = self.client.generate(
                model=self.model,
                prompt="Test",
                stream=False,
            )
            return response is not None
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
