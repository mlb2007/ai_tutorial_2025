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

from dspy.evaluate import Evaluate

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
    background: str = Field(description="Background section of the patent (including 'BACKGROUND' and 'BACKGROUND OF THE INVENTION' sections)")
    claims: List[str] = Field(description="List of patent claims")
    summary: str = Field(description="Concise summary of the patent's key innovations and technical contributions")

class PatentParser(dspy.Signature):
    """Extract structured patent information from HTML content"""
    html: str = dspy.InputField(desc="Raw HTML content from Google Patents page")
    patent_data: PatentData = dspy.OutputField(desc="Structured patent data including background sections but excluding figures and figure explanations")

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
        summary_lower = patent_data.summary.lower()
        title_words = set(patent_data.title.lower().split()[:3])
        abstract_words = set(patent_data.abstract.lower().split()[:5])
        has_title_connection = any(word in summary_lower for word in title_words)
        has_abstract_connection = any(word in summary_lower for word in abstract_words)
        if not (has_title_connection and has_abstract_connection):
            return ValidationResult(False, "Summary lacks connection to title or abstract")
        return ValidationResult(True)

    @staticmethod
    def validate_background(patent_data: PatentData) -> ValidationResult:
        """Validate that background section is properly extracted"""
        if not patent_data.background or len(patent_data.background.strip()) < 100:
            return ValidationResult(False, "Background section too short or empty")
        background_lower = patent_data.background.lower()
        background_keywords = ['background', 'invention', 'prior', 'art', 'field', 'technology']
        has_background_keywords = any(keyword in background_lower for keyword in background_keywords)
        if not has_background_keywords:
            return ValidationResult(False, "Background section lacks typical background content")
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
        def has_redundant_claims(claims: List[str]) -> bool:
            def check_pair(claim_pair: tuple) -> bool:
                claim1, claim2 = claim_pair
                return PatentValidator.calculate_similarity(claim1, claim2) > similarity_threshold
            claim_pairs = [(claims[i], claims[j]) 
                          for i in range(len(claims)) 
                          for j in range(i + 1, len(claims))]
            return any(map(check_pair, claim_pairs))
        if has_redundant_claims(patent_data.claims):
            return ValidationResult(False, "Claims are too similar")
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
            PatentValidator.validate_background,
            lambda pd: PatentValidator.validate_claims(pd, 0.7)
        ]
        return list(map(lambda validator: validator(patent_data), validators))

def clean_text(text: str) -> str:
    """Clean text by removing figure references and redundant content"""
    if not text:
        return ""
    cleaning_operations = [
        lambda t: re.sub(r'FIG\.?\s*\d+[-\d]*', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'Figure\s*\d+[-\d]*', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'FIGS\.?\s*\d+[-\d]*', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'as\s+(?:shown|illustrated)\s+in\s+(?:FIG\.?|Figure|FIGS\.?)\s*\d+[-\d]*', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'FIG\.?\s*\d+[-\d]*\s*(?:shows|illustrates|depicts|is\s+a\s+view\s+of).*?(?=\n|\.|$)', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'Figure\s*\d+[-\d]*\s*(?:shows|illustrates|depicts|is\s+a\s+view\s+of).*?(?=\n|\.|$)', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'FIGS\.?\s*\d+[-\d]*\s*(?:show|illustrate|depict|are\s+views\s+of).*?(?=\n|\.|$)', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'Referring\s+to\s+(?:FIG\.?|Figure|FIGS\.?)\s*\d+[-\d]*.*?(?=\n\n|\n[A-Z]|$)', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'In\s+(?:FIG\.?|Figure|FIGS\.?)\s*\d+[-\d]*.*?(?=\n\n|\n[A-Z]|$)', '', t, flags=re.IGNORECASE),
        lambda t: re.sub(r'\s+', ' ', t),
        lambda t: t.strip()
    ]
    return reduce(lambda text, operation: operation(text), cleaning_operations, text)


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

# Define a signature for the judge
class ExtractionJudge(dspy.Signature):
    html: str = dspy.InputField(desc="Original HTML content")
    extracted: dict = dspy.InputField(desc="Extracted fields")
    assessment: bool = dspy.OutputField(desc="Is the extraction correct and comprehensive?")

def create_extraction_metric():
    """Create a closure that encapsulates the judge instance for extraction evaluation"""
    # Create a judge module, asking the LLM itself to evaluate itself
    judge = dspy.Predict(ExtractionJudge)
    
    def dspy_extraction_metric(example, pred, trace=None):
        assessment = judge(html=example.html, extracted=pred).assessment # should be True/False
        validation_results = PatentValidator.validate_all(pred.patent_data)
        if all(vr.is_valid for vr in validation_results):
            return True & assessment
        failed_validations = [vr for vr in validation_results if not vr.is_valid]
        print(f"Validation failures: {[vr.message for vr in failed_validations]}, LLM assessment: {assessment}")
        return False & assessment
    
    return dspy_extraction_metric

# Create the metric function using the closure
dspy_extraction_metric = create_extraction_metric()

def extract_patent_data(patent_url: str) -> ExtractionResult:
    """Main function to extract structured patent data with cost tracking"""
    print(f"Fetching the HTML content from the patent URL: {patent_url}")
    html_content = fetch_patent_html(patent_url)

    print(f"Predicting the patent data from the HTML content using DSPy prompt")
    predictor = dspy.Predict(PatentParser)
    result = predictor(html=html_content)

    print(f"Evaluating the patent data using the LLM-as-judge metric")
    # Evaluate using the LLM-as-judge metric, with_inputs() is essential.
    devset = [dspy.Example(html=html_content).with_inputs('html')]
    evaluator = dspy.evaluate.Evaluate(devset=devset, metric=create_extraction_metric())
    eval_results = evaluator(predictor)
    
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

    total_cost = calculate_total_cost(lm)

    return cleaned_data, total_cost, eval_results

def process_patents_batch(patent_ids: List[tuple[str, int]]) -> dict:
    """
    Process a batch of patent IDs and return structured data keyed by patent ID.
    Args:
        patent_ids: List of tuples containing (country_code, patent_number)
    Returns:
        Dictionary with patent IDs as keys and tuples of (patent_data, total_cost, attempts) as values
    """
    def construct_patent_url(country_code: str, patent_number: int) -> str:
        return f"https://patents.google.com/patent/{country_code}{patent_number}"
    def extract_patent_id(country_code: str, patent_number: int) -> str:
        return f"{country_code}{patent_number}"
    def process_single_patent(patent_tuple: tuple[str, int]) -> tuple[str, tuple]:
        country_code, patent_number = patent_tuple
        try:
            patent_url = construct_patent_url(country_code, patent_number)
            patent_data, total_cost, eval_results = extract_patent_data(patent_url)
            patent_id = extract_patent_id(country_code, patent_number)
            return patent_id, (patent_data, total_cost, eval_results)
        except Exception as e:
            patent_id = extract_patent_id(country_code, patent_number)
            print(f"Error processing patent {patent_id}: {e}")
            return patent_id, (None, 0.0)
    results = dict(map(process_single_patent, patent_ids))
    successful_extractions = sum(1 for data in results.values() if data[0] is not None)
    total_cost = sum(data[1] for data in results.values() if data[0] is not None)
    print(f"Batch processing complete:")
    print(f"  Successful extractions: {successful_extractions}/{len(patent_ids)}")
    print(f"  Total cost: ${total_cost:.6f}")
    return results

# Example usage
if __name__ == "__main__":
    patent_ids = [
        ("US", 6853284)
    ]
    try:
        patent_results = process_patents_batch(patent_ids)
        for patent_id, (patent_data, total_cost, eval_results) in patent_results.items():
            if patent_data is not None:
                print(f"\n=== Patent {patent_id} ===")
                print("Title:", patent_data.title)
                print("Abstract:", patent_data.abstract[:200] + "..." if len(patent_data.abstract) > 200 else patent_data.abstract)
                print("Background:", patent_data.background[:200] + "..." if len(patent_data.background) > 200 else patent_data.background)
                print("Claims:")
                for _, claim in enumerate(patent_data.claims, 1):
                    print(f"  {claim}")
                print("Summary:", patent_data.summary)
                print(f"Claims count: {len(patent_data.claims)}")
                print(f"Cost: ${total_cost:.6f}")
                print(f"Evaluation results: {eval_results}")
            else:
                print(f"\n=== Patent {patent_id} ===")
                print("Failed to extract data")
    except Exception as e:
        print(f"Error: {e}")