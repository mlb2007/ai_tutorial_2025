import dspy
from pydantic import BaseModel, Field
from typing import List, Optional, Callable
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import re
from functools import reduce
from dataclasses import dataclass

# Suppress Pydantic warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# Load environment variables
load_dotenv()

# Get API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY_PERSONAL')

# Configure DSPy with OpenAI
lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
dspy.configure(lm=lm)

class PatentData(BaseModel):
    """Structured patent data extracted from Google Patents"""
    title: str = Field(description="Patent title")
    abstract: str = Field(description="Patent abstract")
    background: str = Field(description="Background section of the patent")
    claims: List[str] = Field(description="List of patent claims")
    summary: str = Field(description="Concise summary of the patent's key innovations and technical contributions")

class PatentParser(dspy.Signature):
    """Extract structured patent information from HTML content"""
    
    html_content: str = dspy.InputField(desc="Raw HTML content from Google Patents page")
    
    patent_data: PatentData = dspy.OutputField(desc="Structured patent data")

@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    message: str = ""

@dataclass
class ExtractionResult:
    """Result of patent extraction with cost tracking"""
    patent_data: PatentData
    total_cost: float
    attempts: int

class PatentValidator:
    """Functional validator for patent data using composition pattern"""
    
    @staticmethod
    def validate_summary(patent_data: PatentData) -> ValidationResult:
        """Validate that the summary captures key aspects of the patent"""
        if not patent_data.summary or len(patent_data.summary.strip()) < 50:
            return ValidationResult(False, "Summary too short or empty")
        
        # Check if summary mentions key components using functional approach
        summary_lower = patent_data.summary.lower()
        title_words = set(patent_data.title.lower().split()[:3])
        abstract_words = set(patent_data.abstract.lower().split()[:5])
        
        has_title_connection = any(word in summary_lower for word in title_words)
        has_abstract_connection = any(word in summary_lower for word in abstract_words)
        
        if not (has_title_connection and has_abstract_connection):
            return ValidationResult(False, "Summary lacks connection to title or abstract")
        
        return ValidationResult(True)
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    @staticmethod
    def validate_claims(patent_data: PatentData, similarity_threshold: float = 0.7) -> ValidationResult:
        """Validate that claims are non-redundant and well-structured"""
        if not patent_data.claims:
            return ValidationResult(False, "No claims found")
        
        # Check for redundancy among claims using functional approach
        def has_redundant_claims(claims: List[str]) -> bool:
            def check_pair(claim_pair: tuple) -> bool:
                claim1, claim2 = claim_pair
                return PatentValidator.calculate_similarity(claim1, claim2) > similarity_threshold
            
            # Generate all pairs of claims
            claim_pairs = [(claims[i], claims[j]) 
                          for i in range(len(claims)) 
                          for j in range(i + 1, len(claims))]
            
            return any(map(check_pair, claim_pairs))
        
        if has_redundant_claims(patent_data.claims):
            return ValidationResult(False, "Claims are too similar")
        
        # Check that claims are properly cleaned (no figure references)
        figure_patterns = [
            r'FIG\.?\s*\d+[-\d]*',
            r'Figure\s*\d+[-\d]*',
            r'FIGS\.?\s*\d+[-\d]*'
        ]
        
        def has_figure_references(claim: str) -> bool:
            return any(re.search(pattern, claim, flags=re.IGNORECASE) 
                      for pattern in figure_patterns)
        
        if any(map(has_figure_references, patent_data.claims)):
            return ValidationResult(False, "Claims contain figure references")
        
        return ValidationResult(True)
    
    @staticmethod
    def validate_all(patent_data: PatentData) -> List[ValidationResult]:
        """Apply all validations and return results"""
        validators = [
            PatentValidator.validate_summary,
            lambda pd: PatentValidator.validate_claims(pd, 0.7)
        ]
        
        return list(map(lambda validator: validator(patent_data), validators))

def clean_text(text: str) -> str:
    """Clean text by removing figure references and redundant content"""
    if not text:
        return ""
    
    # Define cleaning operations as pure functions
    cleaning_operations = [
        lambda t: re.sub(r'FIG\.?\s*\d+[-\d]*', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'Figure\s*\d+[-\d]*', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'FIGS\.?\s*\d+[-\d]*', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'as\s+(?:shown|illustrated)\s+in\s+(?:FIG\.?|Figure|FIGS\.?)\s*\d+[-\d]*', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'\s+', ' ', t),
        lambda t: t.strip()
    ]
    
    # Apply all cleaning operations using functional composition
    return reduce(lambda text, operation: operation(text), cleaning_operations, text)

def clean_claim_text(claim: str) -> str:
    """Clean claim text by removing redundant claim numbers and extra formatting"""
    if not claim:
        return ""
    
    # Remove redundant claim numbers and formatting patterns
    claim_cleaning_operations = [
        # Remove patterns like "2. 2." or "1. 1." (redundant numbers)
        lambda t: re.sub(r'^(\d+)\.\s*\1\.\s*', r'\1. ', t),
        # Remove patterns like "Claim 1: 1." (redundant with claim prefix)
        lambda t: re.sub(r'^(?:Claim\s*)?(\d+):\s*\1\.\s*', r'\1. ', t, flags=re.IGNORECASE),
        # Remove standard claim number patterns
        lambda t: re.sub(r'^(?:Claim\s*)?\d+\.?\s*', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'^\d+\)\s*', '', t),
        # Clean up multiple spaces and trim
        lambda t: re.sub(r'\s+', ' ', t),
        lambda t: t.strip()
    ]
    
    return reduce(lambda text, operation: operation(text), claim_cleaning_operations, claim)

def consolidate_claims(claims: List[str]) -> List[str]:
    """Consolidate claims by removing redundant numbering and merging related claims"""
    if not claims:
        return []
    
    consolidated = []
    current_claim_parts = []
    
    for claim in claims:
        cleaned_claim = clean_claim_text(claim)
        if cleaned_claim:
            # Check if this claim references a previous claim (e.g., "according to claim 1")
            if re.search(r'according\s+to\s+claim\s+\d+', cleaned_claim, flags=re.IGNORECASE):
                # This is a dependent claim, add to current parts
                current_claim_parts.append(cleaned_claim)
            else:
                # This is an independent claim, finalize previous and start new
                if current_claim_parts:
                    consolidated.append(' '.join(current_claim_parts))
                    current_claim_parts = []
                current_claim_parts = [cleaned_claim]
    
    # Add the last group of claims
    if current_claim_parts:
        consolidated.append(' '.join(current_claim_parts))
    
    return consolidated

def fetch_patent_html(patent_url: str) -> str:
    """Fetch HTML content from Google Patents URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(patent_url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch patent data: {e}")

def calculate_total_cost(lm: dspy.LM) -> float:
    """Calculate total cost from LM history using functional approach"""
    return sum([x['cost'] for x in lm.history if x['cost'] is not None])

def extract_patent_data(patent_url: str) -> ExtractionResult:
    """Main function to extract structured patent data with cost tracking"""
    
    # Fetch HTML content
    html_content = fetch_patent_html(patent_url)
    
    # Create DSPy predictor
    predictor = dspy.Predict(PatentParser)
    
    # Extract structured data
    result = predictor(html_content=html_content)
    
    # Validate and potentially regenerate if validations fail
    max_retries = 3
    attempts = 1
    
    for attempt in range(max_retries):
        validation_results = PatentValidator.validate_all(result.patent_data)
        
        # Check if all validations pass
        if all(vr.is_valid for vr in validation_results):
            break
        
        # Log validation failures
        failed_validations = [vr for vr in validation_results if not vr.is_valid]
        print(f"Attempt {attempt + 1}: Validation failures: {[vr.message for vr in failed_validations]}")
        
        # Retry with more specific instructions
        result = predictor(html_content=html_content)
        attempts += 1
    
    # Calculate total cost
    total_cost = calculate_total_cost(lm)
    
    # Clean and process the extracted data using functional approach
    def clean_patent_field(field_value: str) -> str:
        return clean_text(field_value)
    
    def clean_claims(claims: List[str]) -> List[str]:
        return list(filter(None, map(clean_patent_field, claims)))
    
    cleaned_data = PatentData(
        title=clean_patent_field(result.patent_data.title),
        abstract=clean_patent_field(result.patent_data.abstract),
        background=clean_patent_field(result.patent_data.background),
        claims=clean_claims(result.patent_data.claims),
        summary=clean_patent_field(result.patent_data.summary)
    )
    
    return cleaned_data

# Example usage
if __name__ == "__main__":
    patent_url = "https://patents.google.com/patent/US6853284"

    try:
        patent_data = extract_patent_data(patent_url)
        print("Patent Title:", patent_data.title)
        print("Abstract:", patent_data.abstract)
        print("Background:", patent_data.background)
        print("Summary:", patent_data.summary)
        print("Claims:")
        for i, claim in enumerate(patent_data.claims, 1):
            print(f"  {i}. {claim}")
    except Exception as e:
        print(f"Error: {e}")