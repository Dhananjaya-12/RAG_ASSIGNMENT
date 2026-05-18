"""
LLM Generator Module
Handles response generation using either Google Gemini API or a mock generator.
"""

import os
from typing import List, Dict, Optional


class GeminiGenerator:
    """
    LLM generator using Google's Gemini API.
    Requires GOOGLE_API_KEY environment variable.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature
        self._model = None
    
    @property
    def model(self):
        """Lazy-load the Gemini model."""
        if self._model is None:
            import google.generativeai as genai
            
            api_key = os.getenv("GOOGLE_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not set. Get a free key at https://aistudio.google.com/apikey"
                )
            
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": 1024,
                }
            )
            print(f"  Gemini model initialized: {self.model_name}")
        return self._model
    
    def generate(self, prompt: str) -> Dict:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The formatted prompt with context and question
        
        Returns:
            Dict with 'text', 'model', and 'tokens' (estimated)
        """
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            
            return {
                "text": text,
                "model": self.model_name,
                "provider": "gemini",
                "tokens_estimated": len(text.split()) * 1.3,  # rough estimate
            }
        except Exception as e:
            return {
                "text": f"[Generation Error] {str(e)}",
                "model": self.model_name,
                "provider": "gemini",
                "error": str(e),
            }


class MockGenerator:
    """
    Mock generator that creates responses from retrieved context.
    No API key required — useful for testing and development.
    """
    
    def __init__(self):
        self.model_name = "mock-generator"
    
    def generate(self, prompt: str) -> Dict:
        """
        Generate a mock response by extracting and summarizing context.
        
        Extracts key sentences from the provided context to form a coherent answer.
        """
        # Extract context and question from the prompt
        context = ""
        question = ""
        
        if "Context:" in prompt and "Question:" in prompt:
            parts = prompt.split("Question:")
            context_part = parts[0].split("Context:")[-1].strip()
            question = parts[-1].strip().split("\n")[0].strip()
            context = context_part
        else:
            context = prompt
        
        # Extract meaningful sentences from context
        sentences = [s.strip() for s in context.replace('\n', ' ').split('.') if len(s.strip()) > 30]
        
        if not sentences:
            response_text = "Based on the available context, I don't have enough information to provide a detailed answer to this question."
        else:
            # Select top sentences (up to 5) that might be relevant
            selected = sentences[:min(5, len(sentences))]
            response_text = (
                f"Based on the provided context, here is what I found:\n\n"
                + ". ".join(selected) + ".\n\n"
                + "This information is synthesized from the retrieved documents."
            )
        
        return {
            "text": response_text,
            "model": "mock-generator",
            "provider": "mock",
            "tokens_estimated": len(response_text.split()),
        }


def get_generator(provider: str = "gemini", **kwargs):
    """
    Factory function to create the appropriate generator.
    
    Args:
        provider: "gemini" or "mock"
    
    Returns:
        Generator instance
    """
    if provider == "gemini":
        return GeminiGenerator(**kwargs)
    elif provider == "mock":
        return MockGenerator()
    else:
        print(f"  [WARNING] Unknown provider '{provider}', falling back to mock.")
        return MockGenerator()


def format_prompt(template: str, context: str, question: str) -> str:
    """Format the RAG prompt with context and question."""
    return template.format(context=context, question=question)
