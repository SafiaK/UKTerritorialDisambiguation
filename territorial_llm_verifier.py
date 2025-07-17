import re
import json
from typing import Dict, List, Any, Tuple

def clean_text(text: str) -> str:
    """
    Clean text by removing special characters, ellipsis, and normalizing whitespace.
    
    Args:
        text: Input text to clean
    
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove ellipsis
    text = text.replace("...", "")
    
    # Remove special characters but keep alphanumeric, spaces, and basic punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\-\(\)]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text.lower()

def validate_territory_in_section(territory: str, section_text: str) -> bool:
    """
    Check if a territory name appears in the section text.
    
    Args:
        territory: Territory name (e.g., "England", "Wales")
        section_text: Full text of the section
    
    Returns:
        Boolean indicating if territory is found in section
    """
    if not territory or not section_text:
        return False
    
    # Clean both texts
    clean_territory = clean_text(territory)
    clean_section = clean_text(section_text)
    
    # Check if territory appears in section text
    return clean_territory in clean_section

def validate_territory_text_in_section(territory_text: str, section_text: str) -> Tuple[bool, float]:
    """
    Check if the extracted territory text is actually present in the section.
    
    Args:
        territory_text: Extracted text snippet for a territory
        section_text: Full text of the section
    
    Returns:
        Tuple of (is_present: bool, similarity_score: float)
    """
    if not territory_text or not section_text:
        return False, 0.0
    
    # Clean both texts
    clean_territory_text = clean_text(territory_text)
    clean_section_text = clean_text(section_text)
    
    # Remove section number from territory text if present (e.g., "5 Duty to..." -> "Duty to...")
    clean_territory_text = re.sub(r'^\d+\s+', '', clean_territory_text)
    
    # Check if the cleaned territory text is substring of section text
    is_present = clean_territory_text in clean_section_text
    
    # Calculate similarity score (percentage of territory text found in section)
    if is_present:
        similarity_score = 1.0
    else:
        # For partial matches, count overlapping words
        territory_words = set(clean_territory_text.split())
        section_words = set(clean_section_text.split())
        
        if territory_words:
            overlap = len(territory_words.intersection(section_words))
            similarity_score = overlap / len(territory_words)
        else:
            similarity_score = 0.0
    
    return is_present, similarity_score

def validate_territorial_extraction(extracted_data: List[Dict], section_texts: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate territorial extraction results against actual section texts.
    
    Args:
        extracted_data: List of territorial differences with provisions and territory_texts
        section_texts: Dictionary mapping section IDs to their full text content
    
    Returns:
        Comprehensive validation report
    """
    validation_report = {
        "total_differences": len(extracted_data),
        "total_sections_checked": 0,
        "total_territories_checked": 0,
        "territory_presence_validation": {
            "correct": 0,
            "incorrect": 0,
            "missing_sections": []
        },
        "territory_text_validation": {
            "exact_matches": 0,
            "partial_matches": 0,
            "no_matches": 0,
            "average_similarity_score": 0.0
        },
        "detailed_results": [],
        "overall_score": 0.0
    }
    
    all_similarity_scores = []
    
    for diff_idx, difference in enumerate(extracted_data):
        diff_result = {
            "difference_id": difference.get("difference_id", f"diff_{diff_idx}"),
            "provisions": difference.get("provisions", {}),
            "territory_texts": difference.get("territory_texts", {}),
            "validation_results": {}
        }
        
        # Validate each provision (section-territory mapping)
        for section_id, territories in difference.get("provisions", {}).items():
            validation_report["total_sections_checked"] += 1
            
            section_text = section_texts.get(section_id, "")
            
            if not section_text:
                validation_report["territory_presence_validation"]["missing_sections"].append(section_id)
                continue
            
            section_validation = {
                "section_text_available": True,
                "territory_validations": {}
            }
            
            # Check each territory for this section
            for territory in territories:
                validation_report["total_territories_checked"] += 1
                
                # Validate territory presence in section
                territory_present = validate_territory_in_section(territory, section_text)
                
                if territory_present:
                    validation_report["territory_presence_validation"]["correct"] += 1
                else:
                    validation_report["territory_presence_validation"]["incorrect"] += 1
                
                # Validate territory text if available
                territory_text = difference.get("territory_texts", {}).get(territory, "")
                text_present = False
                similarity_score = 0.0
                
                if territory_text:
                    text_present, similarity_score = validate_territory_text_in_section(territory_text, section_text)
                    all_similarity_scores.append(similarity_score)
                    
                    if similarity_score >= 1.0:
                        validation_report["territory_text_validation"]["exact_matches"] += 1
                    elif similarity_score > 0.0:
                        validation_report["territory_text_validation"]["partial_matches"] += 1
                    else:
                        validation_report["territory_text_validation"]["no_matches"] += 1
                
                section_validation["territory_validations"][territory] = {
                    "territory_present_in_section": territory_present,
                    "territory_text_present": text_present,
                    "similarity_score": similarity_score,
                    "territory_text": territory_text
                }
            
            diff_result["validation_results"][section_id] = section_validation
        
        validation_report["detailed_results"].append(diff_result)
    
    # Calculate overall scores
    if validation_report["total_territories_checked"] > 0:
        territory_accuracy = validation_report["territory_presence_validation"]["correct"] / validation_report["total_territories_checked"]
    else:
        territory_accuracy = 0.0
    
    if all_similarity_scores:
        avg_similarity = sum(all_similarity_scores) / len(all_similarity_scores)
        validation_report["territory_text_validation"]["average_similarity_score"] = avg_similarity
    else:
        avg_similarity = 0.0
    
    # Overall score combines territory presence accuracy and text similarity
    validation_report["overall_score"] = (territory_accuracy + avg_similarity) / 2
    
    return validation_report

def validate_from_json_files(extracted_results_path: str, section_texts_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Validate territorial extraction results from JSON files.
    
    Args:
        extracted_results_path: Path to JSON file with extracted territorial differences
        section_texts_path: Path to JSON file with section texts (list of objects with 'id' and 'text')
        output_path: Optional path to save validation report
    
    Returns:
        Validation report dictionary
    """
    # Load extracted results
    with open(extracted_results_path, 'r', encoding='utf-8') as f:
        extracted_data = json.load(f)
    
    # Load section texts
    with open(section_texts_path, 'r', encoding='utf-8') as f:
        section_texts_list = json.load(f)
    
    # Convert section texts from list format to dictionary
    # Input format: [{"id": "section-83", "text": "83 Non-funding..."}, ...]
    # Output format: {"section-83": "83 Non-funding...", ...}
    section_texts = {}
    for section_obj in section_texts_list:
        section_id = section_obj.get("id")
        section_text = section_obj.get("text", "")
        if section_id:
            section_texts[section_id] = section_text
    
    print(f"📚 Loaded {len(section_texts)} sections from {section_texts_path}")
    
    # Run validation
    validation_report = validate_territorial_extraction(extracted_data, section_texts)
    
    # Add file paths to report
    validation_report["input_files"] = {
        "extracted_results": extracted_results_path,
        "section_texts": section_texts_path
    }
    
    # Add section loading info
    validation_report["section_loading_info"] = {
        "total_sections_loaded": len(section_texts),
        "input_format": "list_of_objects",
        "sample_section_ids": list(section_texts.keys())[:5]  # First 5 section IDs as sample
    }
    
    # Save validation report if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        validation_report["output_file"] = output_path
    
    return validation_report

if __name__ == "__main__":
    # Example usage with JSON files
    
    # Validate using JSON files
    validation_report = validate_from_json_files(
        extracted_results_path="data/legislation_sections/Education Act 2005/llm_territorial_differences.json",
        section_texts_path="data/legislation_sections/Education Act 2005/territorial_sections_for_llm.json", 
        output_path="data/legislation_sections/Education Act 2005/validation_report.json"
    )
    
    # Print basic summary
    print("🔍 TERRITORIAL EXTRACTION VALIDATION COMPLETE")
    print(f"📊 Overall Score: {validation_report['overall_score']:.3f}")
    print(f"🎯 Territory Presence Accuracy: {validation_report['territory_presence_validation']['correct']}/{validation_report['territory_presence_validation']['correct'] + validation_report['territory_presence_validation']['incorrect']}")
    print(f"📄 Average Text Similarity: {validation_report['territory_text_validation']['average_similarity_score']:.3f}")
    
    if validation_report.get("output_file"):
        print(f"📁 Detailed report saved to: {validation_report['output_file']}")