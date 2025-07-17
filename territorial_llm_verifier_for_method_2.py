import json
import os
import pandas as pd
from typing import List, Dict, Any, Set
from collections import defaultdict
import re
import sys
from datetime import datetime


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

def validate_territory_text_in_section(territory_text: str, section_text: str) -> tuple[bool, float]:
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
    
    if territory_text in territory_text:
        return True, 1.0
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
class TerritorialLLMVerifierMethod2:
    """
    Verifier for the new Method 2 output structure that groups sections into:
    - territorial_differences: sections with substantive territorial differences
    - other_sections: sections with no substantive differences but territorial mentions
    
    Verification includes:
    1. Coverage: Every input section appears in exactly one output list
    2. Section-Territory mapping accuracy
    3. Text extraction verification for territorial_differences
    4. Comprehensive reporting
    """
    
    def __init__(self, llm_output_path: str, input_sections_path: str ,results_filePath:str):
        """
        Initialize verifier with paths to the LLM output and input sections.
        
        Args:
            llm_output_path: Path to llm_territorial_differences_method2.json
            input_sections_path: Path to territorial_sections_for_llm.json
        """
        self.llm_output_path = llm_output_path
        self.input_sections_path = input_sections_path
        
        # Extract main directory name from the input path
        self.main_directory = os.path.basename(os.path.dirname(input_sections_path))
        
        # Setup results logging
        self.results_file = results_filePath
        os.makedirs('results', exist_ok=True)
        
        # Initialize results logging
        self._log_to_file("="*80)
        self._log_to_file(f"=== TERRITORIAL LLM VERIFIER METHOD 2 EVALUATION ===")
        self._log_to_file(f"Main Directory: {self.main_directory}")
        self._log_to_file(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log_to_file(f"LLM Output Path: {llm_output_path}")
        self._log_to_file(f"Input Sections Path: {input_sections_path}")
        self._log_to_file("="*80)
        
        # Load data
        with open(llm_output_path, 'r', encoding='utf-8') as f:
            self.llm_output = json.load(f)
            
        with open(input_sections_path, 'r', encoding='utf-8') as f:
            self.input_sections = json.load(f)
            
        # Create lookup dictionaries
        self.input_sections_dict = {s['id']: s for s in self.input_sections}
        
        # Initialize scoring
        self.total_score = 0
        self.max_score = 0
        self.verification_details = []
    
    def _log_to_file(self, message: str):
        """
        Log message to both console and results file.
        
        Args:
            message: Message to log
        """
        print(message)
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        
    def verify_all(self) -> Dict[str, Any]:
        """
        Run complete verification and return comprehensive report.
        
        Returns:
            Dict containing verification results, scores, and detailed report
        """
        self._log_to_file("Starting Method 2 LLM Output Verification...")
        self._log_to_file(f"Input sections: {len(self.input_sections)}")
        
        # Parse LLM output structure
        if isinstance(self.llm_output, str):
            # Handle case where output might be a JSON string
            try:
                parsed_output = json.loads(self.llm_output)
            except json.JSONDecodeError:
                return {
                    "error": "LLM output is not valid JSON",
                    "status": "failed"
                }
        else:
            parsed_output = self.llm_output
            
        territorial_diffs = parsed_output.get('territorial_differences', [])
        other_sections = parsed_output.get('other_sections', [])
        
        self._log_to_file(f"Territorial differences groups: {len(territorial_diffs)}")
        self._log_to_file(f"Other sections groups: {len(other_sections)}")
        
        # 1. Verify coverage (each section appears exactly once)
        coverage_result = self._verify_coverage(territorial_diffs, other_sections)
        
        # 2. Verify section-territory mappings
        mapping_result = self._verify_section_territory_mappings(territorial_diffs, other_sections)
        
        # 3. Verify text extraction for territorial differences
        text_extraction_result = self._verify_text_extraction(territorial_diffs)
        
        # 4. Calculate final scores
        final_score = self.total_score
        max_possible = self.max_score
        percentage = (final_score / max_possible * 100) if max_possible > 0 else 0
        
        # 5. Generate comprehensive report
        report = {
            "verification_summary": {
                "total_input_sections": len(self.input_sections),
                "total_score": final_score,
                "max_possible_score": max_possible,
                "percentage": round(percentage, 2),
                "status": "passed" if percentage >= 90 else "needs_review" if percentage >= 70 else "failed"
            },
            "coverage_verification": coverage_result,
            "mapping_verification": mapping_result,
            "text_extraction_verification": text_extraction_result,
            "detailed_findings": self.verification_details
        }
        
        return report
    
    def _verify_coverage(self, territorial_diffs: List[Dict], other_sections: List[Dict]) -> Dict[str, Any]:
        """
        Verify that every input section appears exactly once in the output.
        Awards 1 point per section for correct coverage.
        """
        self._log_to_file("\n1. Verifying Coverage...")
        
        input_section_ids = set(s['id'] for s in self.input_sections)
        output_section_ids = set()
        duplicates = []
        
        # Collect all section IDs from territorial_differences
        for group in territorial_diffs:
            provisions = group.get('provisions', {})
            for section_id in provisions.keys():
                if section_id in output_section_ids:
                    duplicates.append(section_id)
                output_section_ids.add(section_id)
        
        # Collect all section IDs from other_sections
        for group in other_sections:
            provisions = group.get('provisions', {})
            for section_id in provisions.keys():
                if section_id in output_section_ids:
                    duplicates.append(section_id)
                output_section_ids.add(section_id)
        
        # Calculate coverage scores
        missing_sections = input_section_ids - output_section_ids
        extra_sections = output_section_ids - input_section_ids
        correctly_covered = input_section_ids & output_section_ids
        
        # Award points: 1 point per correctly covered section
        coverage_points = len(correctly_covered)
        max_coverage_points = len(input_section_ids)
        
        self.total_score += coverage_points
        self.max_score += max_coverage_points
        
        coverage_result = {
            "points_awarded": coverage_points,
            "max_points": max_coverage_points,
            "percentage": round(coverage_points / max_coverage_points * 100, 2) if max_coverage_points > 0 else 0,
            "correctly_covered": len(correctly_covered),
            "missing_sections": list(missing_sections),
            "extra_sections": list(extra_sections),
            "duplicate_sections": duplicates,
            "total_input_sections": len(input_section_ids),
            "total_output_sections": len(output_section_ids)
        }
        
        # Add to detailed findings
        if missing_sections:
            self.verification_details.append({
                "type": "coverage_error",
                "severity": "high",
                "message": f"Missing sections: {list(missing_sections)}"
            })
        
        if extra_sections:
            self.verification_details.append({
                "type": "coverage_error",
                "severity": "medium",
                "message": f"Extra sections not in input: {list(extra_sections)}"
            })
            
        if duplicates:
            self.verification_details.append({
                "type": "coverage_error",
                "severity": "high",
                "message": f"Duplicate sections: {duplicates}"
            })
        
        self._log_to_file(f"Coverage: {coverage_points}/{max_coverage_points} points ({coverage_result['percentage']}%)")
        
        return coverage_result
    
    def _verify_section_territory_mappings(self, territorial_diffs: List[Dict], other_sections: List[Dict]) -> Dict[str, Any]:
        """
        Verify that section-territory mappings match the input data.
        Awards 1 point per correctly mapped section-territory pair.
        """
        self._log_to_file("\n2. Verifying Section-Territory Mappings...")
        
        correct_mappings = 0
        total_mappings = 0
        mapping_errors = []
        
        # Check territorial_differences
        for group in territorial_diffs:
            provisions = group.get('provisions', {})
            for section_id, territories in provisions.items():
                if section_id in self.input_sections_dict:
                    input_territories = set(self.input_sections_dict[section_id].get('territories', []))
                    output_territories = set(territories)
                    
                    total_mappings += 1
                    
                    if input_territories == output_territories:
                        correct_mappings += 1
                    else:
                        mapping_errors.append({
                            "section_id": section_id,
                            "group_type": "territorial_differences",
                            "input_territories": list(input_territories),
                            "output_territories": list(output_territories),
                            "missing_territories": list(input_territories - output_territories),
                            "extra_territories": list(output_territories - input_territories)
                        })
        
        # Check other_sections
        for group in other_sections:
            provisions = group.get('provisions', {})
            for section_id, territories in provisions.items():
                if section_id in self.input_sections_dict:
                    input_territories = set(self.input_sections_dict[section_id].get('territories', []))
                    output_territories = set(territories)
                    
                    total_mappings += 1
                    
                    if input_territories == output_territories:
                        correct_mappings += 1
                    else:
                        mapping_errors.append({
                            "section_id": section_id,
                            "group_type": "other_sections",
                            "input_territories": list(input_territories),
                            "output_territories": list(output_territories),
                            "missing_territories": list(input_territories - output_territories),
                            "extra_territories": list(output_territories - input_territories)
                        })
        
        # Award points
        self.total_score += correct_mappings
        self.max_score += total_mappings
        
        mapping_result = {
            "points_awarded": correct_mappings,
            "max_points": total_mappings,
            "percentage": round(correct_mappings / total_mappings * 100, 2) if total_mappings > 0 else 0,
            "correct_mappings": correct_mappings,
            "total_mappings": total_mappings,
            "mapping_errors": mapping_errors
        }
        
        # Add significant mapping errors to detailed findings
        for error in mapping_errors:
            self.verification_details.append({
                "type": "mapping_error",
                "severity": "medium",
                "message": f"Section {error['section_id']}: Expected territories {error['input_territories']}, got {error['output_territories']}"
            })
        
        self._log_to_file(f"Mappings: {correct_mappings}/{total_mappings} points ({mapping_result['percentage']}%)")
        
        return mapping_result
    
    def _verify_text_extraction(self, territorial_diffs: List[Dict]) -> Dict[str, Any]:
        """
        Verify that extracted text in territory_texts is actually found in the original section text.
        Awards 1 point per correctly extracted text snippet.
        """
        self._log_to_file("\n3. Verifying Text Extraction...")
        
        correct_extractions = 0
        total_extractions = 0
        extraction_errors = []
        
        for group in territorial_diffs:
            difference_id = group.get('difference_id', 'unknown')
            territory_texts = group.get('territory_texts', {})
            
            for territory, extracted_text in territory_texts.items():
                # Find the relevant section(s) for this territory
                provisions = group.get('provisions', {})
                relevant_sections = [sid for sid, territories in provisions.items() if territory in territories]
                
                total_extractions += 1
                text_found = False
                
                 # Check if extracted text is found in any of the relevant sections using clean_text
                for section_id in relevant_sections:
                    if section_id in self.input_sections_dict:
                        section_text = self.input_sections_dict[section_id]['text']
                        
                        # Use the validate_territory_text_in_section function for robust comparison
                        is_present, sim_score = validate_territory_text_in_section(extracted_text, section_text)
                        
                        if is_present:
                            text_found = True
                            matching_section = section_id
                            similarity_score = sim_score
                            break
                        elif sim_score > similarity_score:
                            # Track the best similarity score even if not exact match
                            similarity_score = sim_score
                            matching_section = section_id
                
                if text_found:
                    correct_extractions += 1
                else:
                    extraction_errors.append({
                        "difference_id": difference_id,
                        "territory": territory,
                        "extracted_text": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
                        "relevant_sections": relevant_sections,
                        "error": "Text not found in any relevant section"
                    })
        
        # Award points
        self.total_score += correct_extractions
        self.max_score += total_extractions
        
        extraction_result = {
            "points_awarded": correct_extractions,
            "max_points": total_extractions,
            "percentage": round(correct_extractions / total_extractions * 100, 2) if total_extractions > 0 else 0,
            "correct_extractions": correct_extractions,
            "total_extractions": total_extractions,
            "extraction_errors": extraction_errors
        }
        
        # Add extraction errors to detailed findings
        for error in extraction_errors:
            self.verification_details.append({
                "type": "extraction_error",
                "severity": "high",
                "message": f"Difference {error['difference_id']}, Territory {error['territory']}: Extracted text not found in original section"
            })
        
        self._log_to_file(f"Text Extraction: {correct_extractions}/{total_extractions} points ({extraction_result['percentage']}%)")
        
        return extraction_result
    
    def write_detailed_excel_report(self, output_path: str = 'results/method2_verification_report.xlsx'):
        """
        Write a comprehensive Excel report with multiple sheets for different aspects of verification.
        """
        self._log_to_file(f"\n4. Writing detailed Excel report to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Sheet 1: Summary
            summary_data = {
                'Metric': ['Total Input Sections', 'Coverage Score', 'Mapping Score', 'Text Extraction Score', 'Overall Score'],
                'Value': [
                    len(self.input_sections),
                    f"{self.total_score}/{self.max_score}",
                    f"Calculated per category",
                    f"Calculated per category", 
                    f"{round(self.total_score/self.max_score*100, 2) if self.max_score > 0 else 0}%"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Section Coverage Details
            coverage_rows = []
            
            # Parse output to track section coverage
            if isinstance(self.llm_output, str):
                try:
                    parsed_output = json.loads(self.llm_output)
                except:
                    parsed_output = {"territorial_differences": [], "other_sections": []}
            else:
                parsed_output = self.llm_output
                
            territorial_diffs = parsed_output.get('territorial_differences', [])
            other_sections = parsed_output.get('other_sections', [])
            
            covered_sections = set()
            
            # Track territorial differences
            for group in territorial_diffs:
                provisions = group.get('provisions', {})
                for section_id, territories in provisions.items():
                    covered_sections.add(section_id)
                    coverage_rows.append({
                        'section_id': section_id,
                        'found_in_output': True,
                        'output_category': 'territorial_differences',
                        'input_territories': ', '.join(self.input_sections_dict.get(section_id, {}).get('territories', [])),
                        'output_territories': ', '.join(territories),
                        'territories_match': set(self.input_sections_dict.get(section_id, {}).get('territories', [])) == set(territories),
                        'difference_type': group.get('difference_type', ''),
                        'what_is_different': group.get('what_is_different', '')
                    })
            
            # Track other sections
            for group in other_sections:
                provisions = group.get('provisions', {})
                for section_id, territories in provisions.items():
                    covered_sections.add(section_id)
                    coverage_rows.append({
                        'section_id': section_id,
                        'found_in_output': True,
                        'output_category': 'other_sections',
                        'input_territories': ', '.join(self.input_sections_dict.get(section_id, {}).get('territories', [])),
                        'output_territories': ', '.join(territories),
                        'territories_match': set(self.input_sections_dict.get(section_id, {}).get('territories', [])) == set(territories),
                        'difference_type': '',
                        'what_is_different': group.get('category', '')
                    })
            
            # Add missing sections
            for section in self.input_sections:
                if section['id'] not in covered_sections:
                    coverage_rows.append({
                        'section_id': section['id'],
                        'found_in_output': False,
                        'output_category': 'MISSING',
                        'input_territories': ', '.join(section.get('territories', [])),
                        'output_territories': '',
                        'territories_match': False,
                        'difference_type': '',
                        'what_is_different': ''
                    })
            
            coverage_df = pd.DataFrame(coverage_rows)
            coverage_df.to_excel(writer, sheet_name='Section_Coverage', index=False)
            
            # Sheet 3: Text Extraction Verification
            extraction_rows = []
            for group in territorial_diffs:
                difference_id = group.get('difference_id', 'unknown')
                territory_texts = group.get('territory_texts', {})
                provisions = group.get('provisions', {})
                
                for territory, extracted_text in territory_texts.items():
                    relevant_sections = [sid for sid, territories in provisions.items() if territory in territories]
                    
                    text_verified = False
                    for section_id in relevant_sections:
                        if section_id in self.input_sections_dict:
                            section_text = self.input_sections_dict[section_id]['text']
                            if extracted_text.strip() in section_text:
                                text_verified = True
                                break
                    
                    extraction_rows.append({
                        'difference_id': difference_id,
                        'territory': territory,
                        'relevant_sections': ', '.join(relevant_sections),
                        'extracted_text_preview': extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text,
                        'text_verified': text_verified,
                        'full_extracted_text': extracted_text
                    })
            
            if extraction_rows:
                extraction_df = pd.DataFrame(extraction_rows)
                extraction_df.to_excel(writer, sheet_name='Text_Extraction', index=False)
            
            # Sheet 4: Detailed Findings
            if self.verification_details:
                findings_df = pd.DataFrame(self.verification_details)
                findings_df.to_excel(writer, sheet_name='Detailed_Findings', index=False)
        
        self._log_to_file(f"Excel report saved: {output_path}")
    
    def print_summary_report(self):
        """Print a concise summary report to console and log to file."""
        verification_result = self.verify_all()
        
        self._log_to_file("\n" + "="*80)
        self._log_to_file("TERRITORIAL LLM VERIFIER METHOD 2 - FINAL REPORT")
        self._log_to_file("="*80)
        
        summary = verification_result['verification_summary']
        self._log_to_file(f"Total Input Sections: {summary['total_input_sections']}")
        self._log_to_file(f"Overall Score: {summary['total_score']}/{summary['max_possible_score']} ({summary['percentage']}%)")
        self._log_to_file(f"Status: {summary['status'].upper()}")
        
        self._log_to_file(f"\nCoverage: {verification_result['coverage_verification']['percentage']}%")
        self._log_to_file(f"Mapping Accuracy: {verification_result['mapping_verification']['percentage']}%")
        self._log_to_file(f"Text Extraction: {verification_result['text_extraction_verification']['percentage']}%")
        
        if verification_result['detailed_findings']:
            self._log_to_file(f"\nIssues Found: {len(verification_result['detailed_findings'])}")
            for finding in verification_result['detailed_findings'][:5]:  # Show first 5
                self._log_to_file(f"  - {finding['severity'].upper()}: {finding['message']}")
            if len(verification_result['detailed_findings']) > 5:
                self._log_to_file(f"  ... and {len(verification_result['detailed_findings']) - 5} more issues")
        
        self._log_to_file("\n" + "="*80)
        self._log_to_file(f"Evaluation completed for directory: {self.main_directory}")
        self._log_to_file(f"Results saved to: {self.results_file}")
        self._log_to_file("="*80)
        
        return verification_result


# Usage example and main function
def main():
    """
    Example usage of the verifier.
    Update paths as needed for your specific files.
    """
    out_put_dir = "data/legislation_sections/Housing and Regeneration Act 2008"
    print(f'{out_put_dir}/llm_territorial_differences_method2.json')
    verifier = TerritorialLLMVerifierMethod2(
        llm_output_path=f'{out_put_dir}/llm_territorial_differences_method2.json',
        input_sections_path=f'{out_put_dir}/territorial_sections_for_llm.json'
    )
    
    # Run verification and get detailed results
    result = verifier.print_summary_report()
    
    # Write detailed Excel report
    verifier.write_detailed_excel_report()
    
    return result


if __name__ == "__main__":
    main()