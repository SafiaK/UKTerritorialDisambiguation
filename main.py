import re
import json
import os
from LegislationHandler import LegislationParser
from typing import List, Dict, Any
from territorial_disambiguation_mapper import filter_sections_with_territorial_mentions
from territorial_llm_extractor import process_sections_llm
import spacy

from dotenv import load_dotenv
from territorial_llm_extractor import process_sections_in_one_go_second_method

from territorial_llm_verifier_for_method_2 import TerritorialLLMVerifierMethod2
load_dotenv()

TERRITORIES = [
    "England", "Wales", "Scotland", "Northern Ireland", "England and Wales", "United Kingdom"
]

def load_sections(act_url: str,output_dir:str) -> List[Dict[str, Any]]:
    """
    Loads all sections from the given Act URL using LegislationHandler.
    Input: act_url (str) - The URL of the Act on legislation.gov.uk
    Output: List of dicts, each with 'id' and 'text' for a section.
    Functionality: Instantiates LegislationParser, calls get_sections(), and returns the list.
    """
    parser = LegislationParser(act_url)
    # parser.save_all_sections_to_files(output_dir)
    return parser.get_sections()


def save_filtered_sections(filtered_sections: List[Dict[str, Any]], output_path: str = "territorial_sections_for_llm.json"):
    """
    Saves the filtered territorial sections to a JSON file for LLM processing.
    Input: filtered_sections (list of dicts), output_path (str, default 'territorial_sections_for_llm.json').
    Output: None (writes file to disk).
    Functionality: Dumps the list of filtered sections (with id, text, territories) to a JSON file for the next pipeline step.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_sections, f, indent=2, ensure_ascii=False)
    print(f"Filtered territorial sections saved to {output_path}")


def save_llm_results(results: List[Dict[str, Any]], output_path: str = "results/llm_territorial_differences.json"):
    """
    Saves the LLM-extracted territorial differences to a JSON file in the results directory.
    Input: results (list of dicts), output_path (str, default 'results/llm_territorial_differences.json').
    Output: None (writes file to disk).
    Functionality: Ensures the results directory exists, then writes the results to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"LLM territorial differences saved to {output_path}")

class TerritorialLLMVerifier:
    """
    Class for verifying LLM outputs for territorial differences.
    Functionality to be implemented: checks that returned text is a substring of the section, format matches schema, etc.
    """
    def __init__(self):
        pass
    # TODO: Implement verification methods


def Load_all_sections(act_url,output_dir):
    """
    Main pipeline function. Loads all sections, filters for territorial mentions using legal-context-aware logic from territorial_disambiguation_mapper, calls the LLM extractor, and saves the result for further processing or verification.
    Input: None (edit act_url below as needed).
    Output: Writes filtered sections and LLM results to JSON files.
    Functionality: Orchestrates the pipeline steps in order, using improved filtering for legal context and LLM extraction.
    """
    print(f"Loading sections from {act_url}")
    sections = load_sections(act_url,output_dir)

    # Save all sections in a list
    all_sections = list(sections)

    # INSERT_YOUR_CODE
    # Save the list of all sections in a file in data/legislation_sections/Education Act 2005
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_sections.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_sections, f, indent=2, ensure_ascii=False)
    print(f"All sections saved to {output_file}")
    # Optionally, call a function to download all sections if such a function exists
    # For example, if you have a function called download_all_sections, you can call it here:
    print(f"Total sections found: {len(sections)}")


def filter_sections(sections_json,output_path):
    """
    Filters sections that mention any UK territory in the context of legal phrases like 'applies to', 'extends to', 'except in', etc.
    """
   
    

    # Read the sections_json file (assume it's a path to a JSON file)
    with open(sections_json, "r", encoding="utf-8") as f:
        sections = json.load(f)

    # Keep only sections where id starts with "section"
    filtered_sections = [s for s in sections if str(s.get("id", "")).startswith("section")]
    # INSERT_YOUR_CODE
    # Save the filtered sections to a JSON file in the same directory as the input

    output_file = sections_json  # Overwrite the input file with filtered data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_sections, f, indent=2, ensure_ascii=False)

    # INSERT_YOUR_CODE
    # Compute and print the number of sections in the filtered list
    print(f"Total number of filtered sections: {len(filtered_sections)}")

    
    nlp = spacy.load("en_core_web_sm")
    filtered = filter_sections_with_territorial_mentions(filtered_sections, nlp=nlp)
    print(f"Sections with territorial legal mentions: {len(filtered)}")
    save_filtered_sections(filtered, output_path)


def run_territorial_disambiguation_pipeline(sections_path, out_put_dir):
    with open(sections_path, "r", encoding="utf-8") as f:
        filtered_sections = json.load(f)
    llm_results = process_sections_in_one_go_second_method(filtered_sections, output_path=out_put_dir)
    results = llm_results.replace("```json", "").replace("```", "")

    # Parse the JSON string into a Python dictionary
    results_dict = json.loads(results)

    # Save the dictionary to a file in the same folder as the input file
    output_json_path = os.path.join(os.path.dirname(sections_path), "llm_territorial_differences_method2.json")
    print(output_json_path)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_json_path}")

if __name__ == "__main__":
    education_act_url = "https://www.legislation.gov.uk/ukpga/2002/32"

    out_put_dir = "data/legislation_sections/Education Act 2002"
    Load_all_sections(education_act_url,out_put_dir ) 

    sections_file_with_territorial_info_path = out_put_dir +"/territorial_sections_for_llm.json"
    
    #Step 2: Filter sections with territorial mentions
    filter_sections(f"{out_put_dir}/all_sections.json",sections_file_with_territorial_info_path)

    output_json_path = f"{out_put_dir}/llm_territorial_differences_method2.json"

    #Step 3: Extract the differences
    run_territorial_disambiguation_pipeline(sections_file_with_territorial_info_path,output_json_path)

    #Step4:Verify the output 
    verifier = TerritorialLLMVerifierMethod2(
        llm_output_path=f'{out_put_dir}/llm_territorial_differences_method2.json',
        input_sections_path=f'{out_put_dir}/territorial_sections_for_llm.json',
        results_filePath='results/results2.txt'
    )
    
    # Run verification and get detailed results
    result = verifier.print_summary_report()
    
    