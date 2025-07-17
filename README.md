# Territorial Disambiguation in UK Legal Data

A robust pipeline for analyzing how UK legislation applies differently across territories (England, Wales, Scotland, Northern Ireland). The project combines XML parsing, NLP, legal context filtering, and LLM-based extraction to identify and summarize territorial differences in legal texts.

## Features
- **Legislation Parsing**: Extracts and processes sections from UK legislation XML.
- **Territorial NLP**: Filters sections mentioning UK territories using legal context-aware rules and spaCy NER.
- **LLM Extraction**: Uses OpenAI GPT-4 to extract and summarize key territorial differences, outputting structured JSON.
- **Verification & Reporting**: Verifies LLM outputs and generates detailed Excel reports for review.
- **Modular Pipeline**: Easily adaptable to different Acts and batch processing.

## Project Structure
```
TerritorialDisambiguation/
├── main.py                        # Main pipeline orchestrator
├── LegislationHandler.py          # XML parsing for legislation
├── territorial_disambiguation_mapper.py  # NLP-based filtering
├── territorial_llm_extractor.py   # LLM extraction logic
├── territorial_llm_verifier_for_method_2.py # Output verification & reporting
├── requirements.txt               # Python dependencies
├── .env                           # API keys and config (not tracked)
├── data/                          # Input/output data (gitignored)
├── results/                       # LLM and verification outputs (gitignored)
└── README.md                      # Project documentation
```

## Setup Instructions
1. **Clone the repository**
   ```sh
   git clone https://github.com/SafiaK/UKTerritorialDisambiguation.git
   cd TerritorialDisambiguation
   ```
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up your .env file**
   - Copy `.env.example` to `.env` (if provided) or create `.env` manually.
   - Add your OpenAI API key and any other required config:
     ```
     OPENAI_API_KEY=sk-...
     ```
4. **Download spaCy model**
   ```sh
   python -m spacy download en_core_web_sm
   ```

## Running the Pipeline
Edit `main.py` or call the main function with your desired Act URL and output directory:

```python
if __name__ == "__main__":
    act_url = "https://www.legislation.gov.uk/ukpga/2002/32"
    out_dir = "data/legislation_sections/Education Act 2002"
    Load_all_sections(act_url, out_dir)
    filter_sections(f"{out_dir}/all_sections.json", f"{out_dir}/territorial_sections_for_llm.json")
    run_territorial_disambiguation_pipeline(f"{out_dir}/territorial_sections_for_llm.json", f"{out_dir}/llm_territorial_differences_method2.json")
    # Verification and reporting steps follow
```

Or run from the command line:
```sh
python main.py
```

## Example Output
- Filtered sections: `data/legislation_sections/<Act>/territorial_sections_for_llm.json`
- LLM differences: `data/legislation_sections/<Act>/llm_territorial_differences_method2.json`
-Verifier Results:`result/results.txt`
