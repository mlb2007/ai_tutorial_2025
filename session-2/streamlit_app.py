import streamlit as st
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Add the session-2 directory to the Python path
sys.path.append(str(Path(__file__).parent / "session-2"))

# Now we can import from parse_patents
try:
    from parse_patents import process_patents_batch, PatentData, ExtractionResult
except ImportError as e:
    st.error(f"Failed to import from parse_patents.py: {e}")
    st.stop()

def parse_patent_ids(raw_text: str, default_prefix: str = "US") -> List[Tuple[str, str]]:
    """Parse raw text input into a list of (country_code, patent_number) tuples."""
    patent_ids = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        
        # Check if line contains only digits (integer ID)
        if line.isdigit():
            patent_ids.append((default_prefix.upper(), line))
        else:
            # Use regex to be more flexible with patent ID formats
            match = re.match(r'([A-Za-z]{2})([A-Za-z0-9]+)', line)
            if match:
                country_code, patent_number = match.groups()
                patent_ids.append((country_code.upper(), patent_number))
            else:
                st.warning(f"Could not parse patent ID: '{line}'. Skipping.")
    return patent_ids

def display_results(results: dict[str, ExtractionResult]):
    """Display the extraction results in a readable format."""
    if not results:
        st.info("No results to display.")
        return

    st.header("Extraction Results")

    for patent_id, result in results.items():
        with st.expander(f"Patent: {patent_id}", expanded=True):
            if result and result.patent_data:
                p_data = result.patent_data
                
                # Display truncated message if present
                if hasattr(result, 'truncated_message') and result.truncated_message:
                    st.warning(result.truncated_message)
                
                # --- Main Information ---
                st.subheader(p_data.title)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Cost", f"${result.total_cost:.6f}")
                with col2:
                    # Assuming eval_results is a list of booleans or a similar structure
                    if hasattr(result, 'eval_results') and result.eval_results:
                        # Simple pass/fail based on the first result for now
                        passed = result.eval_results[0] if isinstance(result.eval_results, list) else result.eval_results
                        st.metric("Evaluation", "✅ Passed" if passed else "❌ Failed")
                    else:
                        st.metric("Evaluation", "N/A")

                st.markdown("---")
                
                # --- Abstract and Summary ---
                if p_data.abstract:
                    st.markdown("#### Abstract")
                    st.markdown(f"> {p_data.abstract.replace('  ', '<br>')}", unsafe_allow_html=True)
                
                if p_data.summary:
                    st.markdown("#### Summary")
                    st.markdown(p_data.summary)
                
                st.markdown("---")

                # --- Background ---
                if p_data.background:
                    st.markdown("#### Background")
                    # Use a text area for potentially long content to keep it clean
                    st.text_area("Background Text", p_data.background, height=400, disabled=True)
                
                # --- Claims ---
                if p_data.claims:
                    st.markdown("#### Claims")
                    claims_text = "\n".join(f"{i}. {claim}" for i, claim in enumerate(p_data.claims, 1))
                    st.text_area("Claims", claims_text, height=500, disabled=True)
                
            else:
                st.error(f"Failed to process patent {patent_id}. Please check the ID and try again.")


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="Patent Analyzer")

    st.title("📄 Patent Data Extractor and Analyzer")
    st.markdown("""
        Enter one or more patent IDs in the text area below, with each ID on a new line. 
        You can enter just the numeric ID (e.g., `6853284`) and it will use the default prefix, 
        or enter the full patent ID (e.g., `US6853284`). The application will fetch, parse, 
        and analyze the patent data using a DSPy pipeline.
    """)

    # --- Input Configuration ---
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # --- Input Area ---
        input_patents = st.text_area(
            "Enter Patent IDs (one per line)", 
            "6853284",
            height=100,
            help="Enter numeric IDs (e.g., 6853284) or full patent IDs (e.g., US6853284)"
        )
    
    with col2:
        # --- Prefix Configuration ---
        default_prefix = st.text_input(
            "Default Prefix",
            value="US",
            max_chars=2,
            help="Default country prefix for numeric patent IDs"
        )

    if st.button("Analyze Patents", type="primary"):
        patent_ids_to_process = parse_patent_ids(input_patents, default_prefix)
        
        if not patent_ids_to_process:
            st.warning("Please enter at least one valid patent ID.")
            return

        with st.spinner("Analyzing patents... This may take a moment."):
            try:
                # We need to modify process_patents_batch to accept strings
                # For now, let's try to make it work by adapting here
                # A better solution is to modify the source script
                
                # The source script has been updated in the user's workspace to accept strings.
                # No adaptation needed here.
                
                results = process_patents_batch(patent_ids_to_process)
                display_results(results)
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")

if __name__ == "__main__":
    main() 