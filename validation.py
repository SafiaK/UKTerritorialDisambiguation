import json
import re
from typing import Dict, List, Any

def quick_validate_sample(key_differences: List[Dict], section_texts: Dict[str, str]):
    """
    Quick validation function for your sample key_differences data.
    
    Args:
        key_differences: Your list of territorial differences
        section_texts: Dict mapping section IDs to their text content
    
    Returns:
        Validation summary with specific issues found
    """
    
    print("🔍 QUICK VALIDATION REPORT")
    print("=" * 50)
    
    total_results = len(key_differences)
    validation_summary = {
        "total_checked": total_results,
        "valid_provisions": 0,
        "territorial_matches": 0,
        "type_consistency": 0,
        "issues_found": []
    }
    
    territory_patterns = {
        "England": r'\bEngland\b',
        "Wales": r'\bWales\b',
        "Scotland": r'\bScotland\b',
        "Northern Ireland": r'\bNorthern Ireland\b'
    }
    
    for i, result in enumerate(key_differences):
        print(f"\n📋 Checking Result {i+1}: {result.get('difference_type', 'Unknown')}")
        
        provisions = result.get("provision", [])
        why_different = result.get("why_is_different", "")
        description = result.get("description", "")
        diff_type = result.get("difference_type", "")
        
        # Check 1: Provision References
        valid_provisions = []
        invalid_provisions = []
        
        for provision in provisions:
            if provision in section_texts:
                valid_provisions.append(provision)
                print(f"   ✅ {provision} - Found in section texts")
            else:
                invalid_provisions.append(provision)
                print(f"   ❌ {provision} - NOT found in section texts")
        
        if len(invalid_provisions) == 0:
            validation_summary["valid_provisions"] += 1
        else:
            validation_summary["issues_found"].append(f"Result {i+1}: Invalid provisions {invalid_provisions}")
        
        # Check 2: Territorial Accuracy
        if valid_provisions:
            # Combine text from valid provisions
            combined_text = " ".join([section_texts[p] for p in valid_provisions])
            
            # Find territories mentioned in the result
            claimed_territories = []
            full_text = f"{why_different} {description}".lower()
            
            for territory, pattern in territory_patterns.items():
                if territory.lower() in full_text:
                    claimed_territories.append(territory)
            
            # Check if claimed territories exist in source
            territorial_match = True
            for territory in claimed_territories:
                if re.search(territory_patterns[territory], combined_text, re.IGNORECASE):
                    print(f"   ✅ Territory '{territory}' found in source text")
                else:
                    print(f"   ❌ Territory '{territory}' claimed but NOT found in source")
                    territorial_match = False
                    validation_summary["issues_found"].append(f"Result {i+1}: Territory '{territory}' not in source")
            
            if territorial_match and claimed_territories:
                validation_summary["territorial_matches"] += 1
            
            # Find territories in source not mentioned in result
            source_territories = []
            for territory, pattern in territory_patterns.items():
                if re.search(pattern, combined_text, re.IGNORECASE):
                    source_territories.append(territory)
            
            missing_territories = set(source_territories) - set(claimed_territories)
            if missing_territories:
                print(f"   ⚠️  Source contains unmentioned territories: {list(missing_territories)}")
        
        # Check 3: Type Consistency
        type_keywords = {
            "scope": ["apply", "applies", "applicable", "jurisdiction", "territory", "specific"],
            "content": ["definition", "different", "varies", "meaning", "defined"],
            "penalty": ["penalty", "penalties", "fine", "punishment", "sanction"],
            "procedure": ["procedure", "process", "method", "manner"],
            "enforcement": ["enforcement", "comply", "implementation"]
        }
        
        if diff_type in type_keywords:
            expected_keywords = type_keywords[diff_type]
            found_keywords = [kw for kw in expected_keywords if kw in description.lower()]
            
            if found_keywords:
                print(f"   ✅ Type '{diff_type}' matches content (found: {found_keywords})")
                validation_summary["type_consistency"] += 1
            else:
                print(f"   ❌ Type '{diff_type}' doesn't match content (expected keywords: {expected_keywords})")
                validation_summary["issues_found"].append(f"Result {i+1}: Type '{diff_type}' inconsistent with description")
    
    # Print Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total Results Checked: {validation_summary['total_checked']}")
    print(f"Valid Provision References: {validation_summary['valid_provisions']}/{total_results} ({validation_summary['valid_provisions']/total_results:.1%})")
    print(f"Territorial Accuracy: {validation_summary['territorial_matches']}/{total_results} ({validation_summary['territorial_matches']/total_results:.1%})")
    print(f"Type Consistency: {validation_summary['type_consistency']}/{total_results} ({validation_summary['type_consistency']/total_results:.1%})")
    
    if validation_summary["issues_found"]:
        print(f"\n🚨 Issues Found ({len(validation_summary['issues_found'])}):")
        for issue in validation_summary["issues_found"]:
            print(f"   • {issue}")
    else:
        print("\n🎉 No major issues found!")
    
    return validation_summary

# Example usage with your sample data
def test_with_sample():
    """Test with your sample key_differences data"""
    
    # Your sample data
    sample_key_differences = [
        {
            "provision": ["section-1", "section-2", "section-5", "section-8", "section-10A", "section-11A"],
            "difference_type": "scope",
            "what_is_different": "Applicability",
            "why_is_different": "England-specific provisions",
            "description": "Sections 1, 2, 5, 8, 10A, and 11A apply exclusively to schools in England, establishing the role and duties of the Chief Inspector of Schools in England."
        },
        {
            "provision": ["section-19", "section-20", "section-23", "section-24", "section-25", "section-27", "section-28", "section-31"],
            "difference_type": "scope",
            "what_is_different": "Applicability",
            "why_is_different": "Wales-specific provisions",
            "description": "Sections 19, 20, 23, 24, 25, 27, 28, and 31 pertain to the Chief Inspector of Education and Training in Wales, outlining his powers and responsibilities specific to Welsh schools."
        },
        {
            "provision": ["section-47", "section-48", "section-50"],
            "difference_type": "content",
            "what_is_different": "Definition of denominational education",
            "why_is_different": "Different educational frameworks in England and Wales",
            "description": "Section 47 defines 'denominational education' for England, while section 50 provides a different definition for Wales, reflecting the distinct educational policies in each territory."
        }
    ]
    
    # Mock section texts (replace with your actual section_texts.json data)
    mock_section_texts = {
        "section-1": "The Chief Inspector of Schools in England shall inspect schools in England...",
        "section-2": "This section applies to schools in England and establishes duties...",
        "section-19": "The Chief Inspector of Education and Training in Wales shall...",
        "section-47": "For the purposes of this Part, 'denominational education' in England means...",
        "section-50": "In Wales, 'denominational education' has a different meaning..."
    }
    
    print("Testing with sample data...")
    print("(Note: Replace mock_section_texts with your actual section_texts.json)")
    
    return quick_validate_sample(sample_key_differences, mock_section_texts)

if __name__ == "__main__":
    # To use with your actual data, do:
    # 
    # # Load your data
    with open("your_section_texts.json", "r") as f:
        section_texts = json.load(f)
    # 
    with open("your_key_differences.json", "r") as f:
        key_differences = json.load(f)["key_differences"]  # Adjust based on your JSON structure
    # 
    # # Run validation
    validation_results = quick_validate_sample(key_differences, section_texts)
    
    # For now, run with sample data
    test_with_sample()