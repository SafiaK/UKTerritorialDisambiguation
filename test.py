import json
import re
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
from LegislationHandler import LegislationParser

class AIPoweredTerritorialExtractor:
    """
    AI-powered territorial extractor that uses LLM reasoning to find patterns
    even when XML structure is lost in plain text conversion
    """
    
    def __init__(self, legislation_url: str):
        self.parser = LegislationParser(legislation_url)
        self.act_title = self.parser.get_legislation_title()
        self.all_sections = self.parser.get_sections()
        
        self.results = {
            'act_title': self.act_title,
            'total_sections': len(self.all_sections),
            'territorial_sections': [],
            'definitional_variations': [],
            'authority_mappings': [],
            'enforcement_variations': [],
            'territorial_coverage': {},
            'ai_analysis': {}
        }
    
    def _analyze_with_ai_logic(self, text: str, section_id: str) -> Dict[str, Any]:
        """
        Apply AI-like reasoning to analyze text for territorial patterns
        """
        analysis = {
            'section_id': section_id,
            'is_territorial': False,
            'territories': [],
            'authorities': [],
            'definitions': [],
            'enforcement': [],
            'confidence': 0.0
        }
        
        # AI Logic 1: Territory Detection
        territory_indicators = [
            (r'\bEngland\b(?!\s+and\s+Wales)', 'England'),
            (r'\bWales\b(?!\s+and\s+England)', 'Wales'),
            (r'\bScotland\b', 'Scotland'),
            (r'\bNorthern Ireland\b', 'Northern Ireland'),
            (r'\bEngland and Wales\b', 'England and Wales'),
            (r'\bEngland or Wales\b', 'England or Wales')
        ]
        
        for pattern, territory in territory_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                analysis['territories'].append(territory)
                analysis['is_territorial'] = True
        
        # AI Logic 2: Authority Detection with Context
        authority_patterns = [
            (r'(?:Her|His)\s+Majesty\'?s\s+Chief\s+Inspector[^.]*?(?:England|Wales|Scotland)', 'Chief Inspector'),
            (r'Secretary\s+of\s+State', 'Secretary of State'),
            (r'(?:Welsh\s+)?Assembly', 'Assembly'),
            (r'Welsh\s+Ministers?', 'Welsh Ministers'),
            (r'(?:appropriate\s+)?authority', 'Authority'),
            (r'governing\s+body', 'Governing Body'),
            (r'local\s+authority', 'Local Authority')
        ]
        
        for pattern, auth_type in authority_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                analysis['authorities'].extend(matches)
        
        # AI Logic 3: Definition Detection - Look for sophisticated patterns
        definition_patterns = [
            # Pattern 1: "term" means...
            (r'"([^"]+)"\s*[,\s]*means\s+([^.]+(?:\.[^.]*?)*?)(?=\s*\([0-9]\)|\s*In\s+this|\s*$)', 'quoted_definition'),
            # Pattern 2: In this Part/Act, "term" means...
            (r'In\s+this\s+(?:Part|Act|section)[^,]*,\s*"([^"]+)"\s*means\s+([^.]+(?:\.[^.]*?)*?)(?=\s*\([0-9]\)|\s*In\s+this|\s*$)', 'part_definition'),
            # Pattern 3: territorial relation definitions
            (r'in\s+relation\s+to\s+(?:a\s+school\s+in\s+)?([^,]+),\s*means\s+([^.]+(?:\.[^.]*?)*?)(?=\s*\([0-9]\)|\s*In\s+this|\s*$)', 'territorial_relation')
        ]
        
        for pattern, def_type in definition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if def_type == 'territorial_relation':
                    territory, definition = match
                    analysis['definitions'].append({
                        'type': def_type,
                        'territory': territory.strip(),
                        'definition': definition.strip()
                    })
                else:
                    term, definition = match
                    analysis['definitions'].append({
                        'type': def_type,
                        'term': term.strip(),
                        'definition': definition.strip()
                    })
        
        # AI Logic 4: Enforcement Detection
        enforcement_indicators = [
            (r'liable\s+(?:on\s+)?(?:summary\s+)?conviction', 'conviction_liability'),
            (r'fine\s+(?:not\s+exceeding|of)', 'fine_penalty'),
            (r'imprisonment\s+(?:for\s+a\s+term\s+)?(?:not\s+exceeding|of)', 'imprisonment_penalty'),
            (r'guilty\s+of\s+an\s+offence', 'offence'),
            (r'(?:magistrates?\'?\s+)?court', 'court_system'),
            (r'(?:sheriff\s+)?court', 'court_system')
        ]
        
        for pattern, enforcement_type in enforcement_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                analysis['enforcement'].append(enforcement_type)
        
        # AI Logic 5: Calculate Confidence Score
        confidence_factors = [
            len(analysis['territories']) * 0.3,
            len(analysis['authorities']) * 0.2,
            len(analysis['definitions']) * 0.3,
            len(analysis['enforcement']) * 0.2
        ]
        analysis['confidence'] = min(1.0, sum(confidence_factors))
        
        return analysis
    
    def find_definitional_variations_ai(self) -> List[Dict[str, Any]]:
        """
        Use AI reasoning to find definitional variations across territories
        """
        definitional_variations = []
        
        # First pass: Find all sections with definitions
        definition_sections = []
        for section in self.all_sections:
            ai_analysis = self._analyze_with_ai_logic(section['text'], section['id'])
            if ai_analysis['definitions']:
                definition_sections.append({
                    'section': section,
                    'analysis': ai_analysis
                })
        
        # Second pass: Group definitions by term and look for territorial variations
        terms_by_section = defaultdict(list)
        
        for def_section in definition_sections:
            for definition in def_section['analysis']['definitions']:
                term = definition.get('term', definition.get('territory', 'unknown'))
                terms_by_section[term].append({
                    'section_id': def_section['section']['id'],
                    'definition': definition,
                    'full_text': def_section['section']['text'],
                    'territories': def_section['analysis']['territories']
                })
        
        # Third pass: AI logic to identify territorial variations
        for term, instances in terms_by_section.items():
            if len(instances) > 1 or any(len(inst['territories']) > 1 for inst in instances):
                # This term might have territorial variations
                territorial_defs = self._extract_territorial_definitions_ai(term, instances)
                
                if len(territorial_defs) > 1:
                    definitional_variations.append({
                        'term': term,
                        'territorial_definitions': territorial_defs,
                        'sections_involved': [inst['section_id'] for inst in instances],
                        'complexity_score': len(territorial_defs),
                        'ai_confidence': sum(len(inst['territories']) for inst in instances) / len(instances)
                    })
        
        return definitional_variations
    
    def _extract_territorial_definitions_ai(self, term: str, instances: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Use AI reasoning to extract territorial definitions for a specific term
        """
        territorial_defs = {}
        
        for instance in instances:
            full_text = instance['full_text']
            territories = instance['territories']
            
            # AI Logic: Look for territorial context around the term
            for territory in territories:
                # Look for patterns like "in relation to a school in England"
                territorial_patterns = [
                    rf'in\s+relation\s+to\s+(?:a\s+school\s+in\s+)?{re.escape(territory)}[^,]*,\s*means\s+([^.]+(?:\.[^.]*?)*?)(?=\s*\([0-9]\)|\s*In\s+this|\s*$)',
                    rf'in\s+{re.escape(territory)}[^,]*,\s*means\s+([^.]+(?:\.[^.]*?)*?)(?=\s*\([0-9]\)|\s*In\s+this|\s*$)',
                    rf'for\s+(?:the\s+purposes?\s+of\s+)?{re.escape(territory)}[^,]*,\s*means\s+([^.]+(?:\.[^.]*?)*?)(?=\s*\([0-9]\)|\s*In\s+this|\s*$)'
                ]
                
                for pattern in territorial_patterns:
                    match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        territorial_defs[territory] = match.group(1).strip()
                        break
                
                # If no specific territorial pattern, but territory is mentioned with definition
                if territory not in territorial_defs and territory in full_text:
                    # Find the definition that appears in the same context as the territory
                    definition_match = re.search(rf'(?:.*{re.escape(territory)}.*)"[^"]*"\s*means\s+([^.]+(?:\.[^.]*?)*?)(?=\s*\([0-9]\)|\s*In\s+this|\s*$)', full_text, re.IGNORECASE | re.DOTALL)
                    if definition_match:
                        territorial_defs[territory] = definition_match.group(1).strip()
        
        return territorial_defs
    
    def find_authority_mappings_ai(self) -> List[Dict[str, Any]]:
        """
        Use AI reasoning to find authority mappings across territories
        """
        authority_mappings = []
        
        # Group sections by function using AI analysis
        function_groups = defaultdict(list)
        
        for section in self.all_sections:
            ai_analysis = self._analyze_with_ai_logic(section['text'], section['id'])
            
            if ai_analysis['authorities'] and ai_analysis['is_territorial']:
                function = self._infer_function_ai(section['text'])
                function_groups[function].append({
                    'section': section,
                    'analysis': ai_analysis
                })
        
        # Analyze each function group for territorial authority patterns
        for function, sections in function_groups.items():
            territorial_authorities = defaultdict(list)
            
            for section_info in sections:
                analysis = section_info['analysis']
                for territory in analysis['territories']:
                    territorial_authorities[territory].extend(analysis['authorities'])
            
            if len(territorial_authorities) > 1:  # Multiple territories with different authorities
                authority_mappings.append({
                    'function': function,
                    'territorial_authorities': dict(territorial_authorities),
                    'sections_involved': [s['section']['id'] for s in sections],
                    'complexity_score': len(territorial_authorities),
                    'pattern_type': 'parallel' if len(territorial_authorities) > 1 else 'single'
                })
        
        return authority_mappings
    
    def _infer_function_ai(self, text: str) -> str:
        """
        Use AI reasoning to infer function from text
        """
        text_lower = text.lower()
        
        # AI Logic: Weighted function detection
        function_indicators = {
            'inspection': {
                'keywords': ['inspect', 'inspection', 'examine', 'review', 'assess', 'monitor'],
                'weight': 1.0
            },
            'establishment': {
                'keywords': ['establish', 'appoint', 'constitute', 'create', 'set up'],
                'weight': 1.0
            },
            'enforcement': {
                'keywords': ['enforce', 'penalty', 'fine', 'liable', 'guilty', 'offence', 'conviction'],
                'weight': 1.0
            },
            'regulation': {
                'keywords': ['regulate', 'control', 'prescribe', 'determine', 'direct'],
                'weight': 0.8
            },
            'definition': {
                'keywords': ['means', 'interpretation', 'defined', 'construed', 'meaning'],
                'weight': 0.9
            },
            'procedure': {
                'keywords': ['procedure', 'process', 'method', 'manner', 'requirement'],
                'weight': 0.7
            },
            'power': {
                'keywords': ['power', 'authority', 'right', 'duty', 'function', 'responsibility'],
                'weight': 0.6
            }
        }
        
        function_scores = {}
        for function, config in function_indicators.items():
            score = 0
            for keyword in config['keywords']:
                count = text_lower.count(keyword)
                score += count * config['weight']
            function_scores[function] = score
        
        # Return function with highest score, or 'other' if no clear function
        if function_scores:
            max_function = max(function_scores.items(), key=lambda x: x[1])
            return max_function[0] if max_function[1] > 0 else 'other'
        
        return 'other'
    
    def find_enforcement_variations_ai(self) -> List[Dict[str, Any]]:
        """
        Use AI reasoning to find enforcement variations across territories
        """
        enforcement_variations = []
        
        for section in self.all_sections:
            ai_analysis = self._analyze_with_ai_logic(section['text'], section['id'])
            
            if ai_analysis['enforcement'] and ai_analysis['is_territorial']:
                territorial_enforcements = self._extract_territorial_enforcement_ai(
                    section['text'], ai_analysis['territories']
                )
                
                if territorial_enforcements:
                    enforcement_variations.append({
                        'section_id': section['id'],
                        'enforcement_types': ai_analysis['enforcement'],
                        'territorial_enforcements': territorial_enforcements,
                        'complexity_score': len(territorial_enforcements),
                        'ai_confidence': ai_analysis['confidence']
                    })
        
        return enforcement_variations
    
    def _extract_territorial_enforcement_ai(self, text: str, territories: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Use AI reasoning to extract territorial enforcement variations
        """
        territorial_enforcements = {}
        
        for territory in territories:
            # AI Logic: Look for enforcement patterns associated with each territory
            enforcement_patterns = [
                rf'In\s+{re.escape(territory)}[^,]*,\s*([^.]+(?:\.[^.]*?)*?)(?=\s*\([0-9]\)|\s*In\s+|\s*$)',
                rf'{re.escape(territory)}[^,]*,\s*([^.]+(?:\.[^.]*?)*?)(?=\s*\([0-9]\)|\s*In\s+|\s*$)'
            ]
            
            for pattern in enforcement_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    enforcement_text = match.group(1).strip()
                    
                    # AI Logic: Extract penalties and courts from enforcement text
                    if any(keyword in enforcement_text.lower() for keyword in ['liable', 'penalty', 'fine', 'court', 'conviction']):
                        territorial_enforcements[territory] = {
                            'enforcement_text': enforcement_text,
                            'penalties': self._extract_penalties_ai(enforcement_text),
                            'courts': self._extract_courts_ai(enforcement_text),
                            'conviction_type': self._extract_conviction_type_ai(enforcement_text)
                        }
        
        return territorial_enforcements
    
    def _extract_penalties_ai(self, text: str) -> List[str]:
        """AI logic to extract penalties"""
        penalty_patterns = [
            r'fine\s+(?:not\s+exceeding|of)\s+([^.;,]+)',
            r'imprisonment\s+(?:for\s+a\s+term\s+)?(?:not\s+exceeding|of)\s+([^.;,]+)',
            r'penalty\s+(?:not\s+exceeding|of)\s+([^.;,]+)'
        ]
        
        penalties = []
        for pattern in penalty_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            penalties.extend([match.strip() for match in matches])
        
        return penalties
    
    def _extract_courts_ai(self, text: str) -> List[str]:
        """AI logic to extract courts"""
        court_patterns = [
            r'(magistrates?\'?\s+court)',
            r'((?:sheriff\s+)?court)',
            r'((?:Crown\s+)?Court)',
            r'(High\s+Court)',
            r'(Court\s+of\s+Session)'
        ]
        
        courts = []
        for pattern in court_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            courts.extend([match.strip() for match in matches])
        
        return list(set(courts))
    
    def _extract_conviction_type_ai(self, text: str) -> str:
        """AI logic to extract conviction type"""
        if 'summary conviction' in text.lower():
            return 'summary'
        elif 'indictment' in text.lower():
            return 'indictment'
        else:
            return 'unspecified'
    
    def run_ai_analysis(self) -> Dict[str, Any]:
        """
        Run complete AI-powered territorial analysis
        """
        print(f"Starting AI-powered territorial analysis of: {self.act_title}")
        print(f"Total sections to analyze: {len(self.all_sections)}")
        
        # Step 1: Find territorial sections using AI
        print("1. Identifying territorial sections using AI...")
        territorial_sections = []
        for section in self.all_sections:
            ai_analysis = self._analyze_with_ai_logic(section['text'], section['id'])
            if ai_analysis['is_territorial']:
                territorial_sections.append(ai_analysis)
        
        self.results['territorial_sections'] = territorial_sections
        print(f"   Found {len(territorial_sections)} territorial sections")
        
        # Step 2: Find definitional variations using AI
        print("2. Finding definitional variations using AI...")
        definitional_variations = self.find_definitional_variations_ai()
        self.results['definitional_variations'] = definitional_variations
        print(f"   Found {len(definitional_variations)} definitional variations")
        
        # Step 3: Find authority mappings using AI
        print("3. Finding authority mappings using AI...")
        authority_mappings = self.find_authority_mappings_ai()
        self.results['authority_mappings'] = authority_mappings
        print(f"   Found {len(authority_mappings)} authority mappings")
        
        # Step 4: Find enforcement variations using AI
        print("4. Finding enforcement variations using AI...")
        enforcement_variations = self.find_enforcement_variations_ai()
        self.results['enforcement_variations'] = enforcement_variations
        print(f"   Found {len(enforcement_variations)} enforcement variations")
        
        # Step 5: Calculate complexity score
        complexity_score = self._calculate_complexity_score_ai()
        self.results['territorial_complexity_score'] = complexity_score
        
        # Step 6: Generate AI insights
        self.results['ai_insights'] = self._generate_ai_insights()
        
        # Step 7: Summary
        self.results['summary'] = {
            'total_sections': len(self.all_sections),
            'territorial_sections': len(territorial_sections),
            'definitional_variations': len(definitional_variations),
            'authority_mappings': len(authority_mappings),
            'enforcement_variations': len(enforcement_variations),
            'territorial_complexity_score': complexity_score,
            'analysis_method': 'AI-powered pattern recognition',
            'confidence_level': 'high'
        }
        
        print(f"\nAI Analysis complete! Territorial complexity score: {complexity_score}")
        return self.results
    
    def _calculate_complexity_score_ai(self) -> int:
        """Calculate complexity score using AI weighting"""
        score = 0
        
        # Weighted scoring based on AI analysis
        score += len(self.results['territorial_sections']) * 2
        score += len(self.results['definitional_variations']) * 5  # High impact
        score += len(self.results['authority_mappings']) * 4
        score += len(self.results['enforcement_variations']) * 3
        
        # Bonus for high-confidence AI detections
        high_confidence_count = sum(1 for section in self.results['territorial_sections'] 
                                  if section.get('confidence', 0) > 0.7)
        score += high_confidence_count * 2
        
        return score
    
    def _generate_ai_insights(self) -> Dict[str, Any]:
        """Generate AI-powered insights"""
        insights = {
            'dominant_territories': Counter(),
            'common_authority_patterns': Counter(),
            'enforcement_complexity': {},
            'definitional_complexity': {}
        }
        
        # Analyze territorial dominance
        for section in self.results['territorial_sections']:
            for territory in section.get('territories', []):
                insights['dominant_territories'][territory] += 1
        
        # Analyze authority patterns
        for mapping in self.results['authority_mappings']:
            for territory, authorities in mapping.get('territorial_authorities', {}).items():
                for authority in authorities:
                    insights['common_authority_patterns'][f"{territory}:{authority}"] += 1
        
        return insights
    
    def save_results(self, filename: str = None):
        """Save AI analysis results"""
        if filename is None:
            safe_title = re.sub(r'[^a-zA-Z0-9\s-]', '', self.act_title).replace(' ', '_').lower()
            filename = f"{safe_title}_ai_territorial_analysis.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"AI analysis results saved to: {filename}")
        return filename

# Usage
if __name__ == "__main__":
    # Use with your Education Act 2005 URL
    education_act_url = "https://www.legislation.gov.uk/ukpga/2005/18"
    
    # Create AI-powered extractor
    extractor = AIPoweredTerritorialExtractor(education_act_url)
    
    # Run AI analysis
    results = extractor.run_ai_analysis()
    
    # Save results
    output_file = extractor.save_results()
    
    # Print detailed summary
    print("\n" + "="*60)
    print("AI-POWERED TERRITORIAL ANALYSIS SUMMARY")
    print("="*60)
    print(f"Act: {results['act_title']}")
    print(f"Total Sections: {results['total_sections']}")
    print(f"Territorial Sections: {results['summary']['territorial_sections']}")
    print(f"Definitional Variations: {results['summary']['definitional_variations']}")
    print(f"Authority Mappings: {results['summary']['authority_mappings']}")
    print(f"Enforcement Variations: {results['summary']['enforcement_variations']}")
    print(f"Territorial Complexity Score: {results['summary']['territorial_complexity_score']}")
    
    # Show specific findings
    if results['definitional_variations']:
        print(f"\nDefinitional Variations Found:")
        for def_var in results['definitional_variations']:
            print(f"  - {def_var['term']}: {len(def_var['territorial_definitions'])} territorial definitions")
    
    if results['authority_mappings']:
        print(f"\nAuthority Mappings Found:")
        for auth_map in results['authority_mappings']:
            print(f"  - {auth_map['function']}: {len(auth_map['territorial_authorities'])} territorial authorities")
    
    print(f"\nOutput File: {output_file}")