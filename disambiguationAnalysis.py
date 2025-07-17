#!/usr/bin/env python3
"""
LegalDocML-Based Territorial Disambiguation NLP Analyzer

This system:
1. Extends the existing LegislationHandler to work with LegalDocML XML
2. Uses NLP to detect territorial disambiguation in XML structure
3. Classifies using academically grounded categories
4. Works with data.akn XML format from legislation.gov.uk
5. No web scraping - only proper XML parsing

Academic Categories:
1. Jurisdictional Authority Conflicts
2. Federal-Devolved Power Distribution  
3. Enforcement Jurisdiction Variations
4. Legislative Territorial Scope Ambiguity
5. Constitutional Competence Conflicts
"""

import xml.etree.ElementTree as ET
import re
import json
import spacy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import requests

# Import the existing LegislationHandler
from LegislationHandler import LegislationParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DisambiguationType(Enum):
    """Academically grounded territorial disambiguation categories"""
    JURISDICTIONAL_AUTHORITY = "jurisdictional_authority_conflicts"
    FEDERAL_DEVOLVED_POWER = "federal_devolved_power_distribution" 
    ENFORCEMENT_VARIATION = "enforcement_jurisdiction_variations"
    TERRITORIAL_SCOPE_AMBIGUITY = "legislative_territorial_scope_ambiguity"
    CONSTITUTIONAL_COMPETENCE = "constitutional_competence_conflicts"
    NO_DISAMBIGUATION = "no_territorial_disambiguation"

@dataclass
class TerritorialEvidence:
    """Evidence for territorial disambiguation found in LegalDocML"""
    xml_element: str  # XML element where evidence found
    text_snippet: str
    pattern_type: str
    confidence: float
    method: str  # 'xml_structure', 'nlp_entity', 'semantic_pattern', 'legal_marker'
    territorial_indicators: List[str] = field(default_factory=list)
    legal_significance: str = ""
    xml_path: str = ""  # XPath to the element

@dataclass
class DisambiguationResult:
    """Complete territorial disambiguation analysis result"""
    act_url: str
    section_analyzed: str
    disambiguation_type: DisambiguationType
    confidence: float
    evidence: List[TerritorialEvidence]
    territories_identified: List[str]
    legal_authorities: List[str]
    xml_elements_flagged: List[str]
    recommendations: List[str]
    academic_classification: str
    suggested_xml_enhancements: List[str]

class LegalDocMLTerritorialAnalyzer(LegislationParser):
    """Enhanced LegislationParser with NLP territorial disambiguation"""
    
    def __init__(self, url, is_section=False):
        # Initialize parent LegislationParser
        super().__init__(url, is_section)
        
        # Initialize NLP components
        self.nlp = self._load_nlp_model()
        self._initialize_academic_patterns()
        self._initialize_territorial_vectorizer()
        
        # Academic classification system
        self.academic_categories = {
            DisambiguationType.JURISDICTIONAL_AUTHORITY: {
                'keywords': ['jurisdiction', 'authority', 'power', 'competent court', 'territorial extent'],
                'xml_elements': ['jurisdiction', 'extent', 'application'],
                'legal_significance': 'Conflicts over which authority has jurisdiction'
            },
            DisambiguationType.FEDERAL_DEVOLVED_POWER: {
                'keywords': ['devolved', 'reserved', 'Scottish Ministers', 'Welsh Ministers', 'Northern Ireland Assembly'],
                'xml_elements': ['extent', 'application', 'territorialExtent'],
                'legal_significance': 'Distribution of powers between UK and devolved governments'
            },
            DisambiguationType.ENFORCEMENT_VARIATION: {
                'keywords': ['enforcement', 'proceedings', 'court', 'appeal', 'penalty', 'prosecution'],
                'xml_elements': ['section', 'subsection'],
                'legal_significance': 'Different enforcement mechanisms across territories'
            },
            DisambiguationType.TERRITORIAL_SCOPE_AMBIGUITY: {
                'keywords': ['applies to', 'extends to', 'in relation to', 'for the purposes of'],
                'xml_elements': ['extent', 'application', 'scope'],
                'legal_significance': 'Unclear territorial application requiring interpretation'
            },
            DisambiguationType.CONSTITUTIONAL_COMPETENCE: {
                'keywords': ['consent', 'Legislative Consent Motion', 'constitutional', 'ultra vires', 'Sewel Convention'],
                'xml_elements': ['preamble', 'extent', 'commencement'],
                'legal_significance': 'Constitutional boundaries and competence conflicts'
            }
        }
    
    def _load_nlp_model(self):
        """Load spaCy NLP model"""
        try:
            nlp = spacy.load("en_core_web_sm")
            # Add custom territorial entities
            if "territorial_ruler" not in nlp.pipe_names:
                territorial_ruler = nlp.add_pipe("entity_ruler", name="territorial_ruler")
                territorial_patterns = [
                    {"label": "TERRITORY", "pattern": "England and Wales"},
                    {"label": "TERRITORY", "pattern": "Scotland"}, 
                    {"label": "TERRITORY", "pattern": "Northern Ireland"},
                    {"label": "TERRITORY", "pattern": "United Kingdom"},
                    {"label": "AUTHORITY", "pattern": "Scottish Ministers"},
                    {"label": "AUTHORITY", "pattern": "Welsh Ministers"},
                    {"label": "AUTHORITY", "pattern": "Secretary of State"},
                    {"label": "COURT", "pattern": "sheriff court"},
                    {"label": "COURT", "pattern": "magistrates' court"},
                    {"label": "COURT", "pattern": "Crown Court"},
                    {"label": "COURT", "pattern": "Sheriff Appeal Court"}
                ]
                territorial_ruler.add_patterns(territorial_patterns)
            return nlp
        except OSError:
            logger.warning("spaCy model not found. NLP features limited.")
            return None
    
    def _initialize_academic_patterns(self):
        """Initialize patterns based on academic legal literature"""
        
        # Jurisdictional Authority patterns (based on territorial jurisdiction doctrine)
        self.jurisdictional_patterns = {
            'territorial_jurisdiction': [
                r'\b(?:court|tribunal|authority)\s+(?:has|have)\s+jurisdiction\b',
                r'\bterritorial\s+(?:jurisdiction|extent|scope)\b',
                r'\b(?:competent|appropriate)\s+(?:court|authority|tribunal)\b'
            ],
            'personal_jurisdiction': [
                r'\bover\s+(?:persons?|individuals?|parties?)\b',
                r'\bin\s+relation\s+to\s+(?:persons?|individuals?)\b'
            ],
            'subject_matter_jurisdiction': [
                r'\b(?:matters?|disputes?|cases?)\s+relating\s+to\b',
                r'\bfor\s+the\s+purposes?\s+of\b'
            ]
        }
        
        # Federal-Devolved Power patterns (based on federalism literature)
        self.federal_devolved_patterns = {
            'reserved_powers': [
                r'\breserved\s+(?:matter|power|function)\b',
                r'\bSecretary\s+of\s+State\b',
                r'\bUK\s+(?:Parliament|Government)\b'
            ],
            'devolved_powers': [
                r'\bdevolved\s+(?:matter|power|function|administration)\b',
                r'\bScottish\s+Ministers?\b',
                r'\bWelsh\s+Ministers?\b',
                r'\bNorthern\s+Ireland\s+(?:Assembly|Executive)\b'
            ],
            'concurrent_powers': [
                r'\bshared\s+(?:responsibility|competence)\b',
                r'\bconcurrent\s+(?:jurisdiction|powers?)\b'
            ]
        }
        
        # Enforcement Variation patterns (based on administrative law)
        self.enforcement_patterns = {
            'enforcement_authority': [
                r'\benforcement.*?(?:is|shall\s+be)\s+the\s+responsibility\s+of\b',
                r'\b(?:may|shall)\s+enforce\b',
                r'\benforcement\s+(?:body|authority|officer)\b'
            ],
            'court_procedures': [
                r'\bproceedings.*?(?:in|before)\s+(?:a\s+)?(?:court|tribunal)\b',
                r'\b(?:magistrates?|sheriff|Crown|County)\s+[Cc]ourt\b',
                r'\bappeals?\s+(?:to|lie\s+to)\b'
            ],
            'penalty_variations': [
                r'\bpenalt(?:y|ies)\b.*?\b(?:court|authority)\b',
                r'\bfine.*?not\s+exceeding\b',
                r'\bsanctions?\b'
            ]
        }
        
        # Territorial Scope Ambiguity patterns (based on statutory interpretation)
        self.scope_ambiguity_patterns = {
            'explicit_territorial': [
                r'\b(?:applies?|extends?)\s+to\s+([^.;]+)\b',
                r'\bin\s+(?:relation\s+to|respect\s+of)\s+([^.;]+)\b',
                r'\bfor\s+(?:the\s+purposes?\s+of|)\s+([^.;]+)\b'
            ],
            'implicit_territorial': [
                r'\bsubject\s+to\s+(?:subsection|paragraph)\b',
                r'\bexcept\s+(?:in|for|where)\b',
                r'\bunless\s+(?:the|)\s*(?:context|circumstances)\b'
            ],
            'territorial_exceptions': [
                r'\b(?:does\s+not|shall\s+not)\s+(?:apply|extend)\s+to\b',
                r'\bexcept\s+(?:in\s+)?(?:relation\s+to|for|where)\b',
                r'\bsave\s+(?:for|where|in\s+respect\s+of)\b'
            ]
        }
        
        # Constitutional Competence patterns (based on constitutional law)
        self.constitutional_patterns = {
            'consent_requirements': [
                r'\bLegislative\s+Consent\s+Motion\b',
                r'\bconsent\s+of\s+(?:the\s+)?(?:Scottish\s+Parliament|Welsh\s+Assembly|Northern\s+Ireland\s+Assembly)\b',
                r'\bSewel\s+Convention\b'
            ],
            'ultra_vires_indicators': [
                r'\bbeyond\s+(?:the\s+)?(?:powers?|competence)\b',
                r'\bwithin\s+(?:the\s+)?(?:competence|powers?)\s+of\b',
                r'\bconstitutional\s+(?:boundaries|limits)\b'
            ],
            'preemption_indicators': [
                r'\bpre-?empts?\b',
                r'\boverrides?\b',
                r'\bnotwithstanding\s+(?:any|)\s*(?:other|previous)\b'
            ]
        }
    
    def _initialize_territorial_vectorizer(self):
        """Initialize TF-IDF vectorizer for semantic territorial analysis"""
        # Training corpus of territorial legal phrases
        territorial_corpus = [
            "applies to England Wales Scotland",
            "enforcement responsibility local authorities", 
            "Scottish Ministers consent required",
            "proceedings before sheriff court",
            "devolved matter Northern Ireland",
            "reserved powers Secretary State",
            "Crown Court appeals England Wales",
            "constitutional competence boundaries",
            "territorial jurisdiction conflicts",
            "Legislative Consent Motion required"
        ]
        
        self.territorial_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=1000,
            stop_words='english'
        )
        self.territorial_vectorizer.fit(territorial_corpus)
    
    def analyze_territorial_disambiguation(self, section_id: Optional[str] = None) -> DisambiguationResult:
        """
        Main method to analyze territorial disambiguation in LegalDocML
        
        Args:
            section_id: Specific section to analyze (e.g., "section-174"), or None for full Act
        """
        logger.info(f"Analyzing territorial disambiguation for {self.url}")
        
        evidence = []
        territories_identified = set()
        authorities_identified = set()
        xml_elements_flagged = []
        
        try:
            # Get the XML tree from parent class
            root = self.tree.getroot()
            
            # Analyze different XML structural levels
            evidence.extend(self._analyze_xml_structure(root, section_id))
            evidence.extend(self._analyze_extent_elements(root))
            evidence.extend(self._analyze_section_content(root, section_id))
            evidence.extend(self._apply_nlp_analysis(root, section_id))
            
            # Extract territories and authorities from evidence
            for ev in evidence:
                territories_identified.update(ev.territorial_indicators)
                if ev.pattern_type in ['authority', 'enforcement']:
                    authorities_identified.add(ev.text_snippet)
                xml_elements_flagged.append(ev.xml_element)
            
            # Classify disambiguation type using academic framework
            disambiguation_type, confidence = self._classify_disambiguation_type(evidence)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(disambiguation_type, evidence)
            
            # Generate XML enhancement suggestions
            xml_enhancements = self._suggest_xml_enhancements(disambiguation_type, evidence)
            
            result = DisambiguationResult(
                act_url=self.url,
                section_analyzed=section_id or "entire_act",
                disambiguation_type=disambiguation_type,
                confidence=confidence,
                evidence=evidence,
                territories_identified=list(territories_identified),
                legal_authorities=list(authorities_identified),
                xml_elements_flagged=list(set(xml_elements_flagged)),
                recommendations=recommendations,
                academic_classification=self._get_academic_classification(disambiguation_type),
                suggested_xml_enhancements=xml_enhancements
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing territorial disambiguation: {e}")
            return DisambiguationResult(
                act_url=self.url,
                section_analyzed=section_id or "entire_act",
                disambiguation_type=DisambiguationType.NO_DISAMBIGUATION,
                confidence=0.0,
                evidence=[],
                territories_identified=[],
                legal_authorities=[],
                xml_elements_flagged=[],
                recommendations=[f"Error in analysis: {str(e)}"],
                academic_classification="Error",
                suggested_xml_enhancements=[]
            )
    
    def _analyze_xml_structure(self, root: ET.Element, section_id: Optional[str]) -> List[TerritorialEvidence]:
        """Analyze XML structure for territorial elements"""
        evidence = []
        
        # Look for extent elements in LegalDocML
        for extent_elem in root.findall(".//akn:extent", self.namespace):
            if extent_elem.text:
                evidence.append(TerritorialEvidence(
                    xml_element="extent",
                    text_snippet=extent_elem.text.strip(),
                    pattern_type="territorial_extent",
                    confidence=0.9,
                    method="xml_structure",
                    territorial_indicators=self._extract_territories_from_text(extent_elem.text),
                    legal_significance="Formal territorial extent declaration",
                    xml_path=self._get_xpath(extent_elem)
                ))
        
        # Look for application elements
        for app_elem in root.findall(".//akn:application", self.namespace):
            if app_elem.text:
                evidence.append(TerritorialEvidence(
                    xml_element="application",
                    text_snippet=app_elem.text.strip(),
                    pattern_type="territorial_application",
                    confidence=0.85,
                    method="xml_structure",
                    territorial_indicators=self._extract_territories_from_text(app_elem.text),
                    legal_significance="Territorial application scope",
                    xml_path=self._get_xpath(app_elem)
                ))
        
        # Look for territorial attributes
        for elem in root.iter():
            if elem.attrib.get('territorial'):
                evidence.append(TerritorialEvidence(
                    xml_element=elem.tag,
                    text_snippet=f"territorial={elem.attrib['territorial']}",
                    pattern_type="territorial_attribute",
                    confidence=0.95,
                    method="xml_structure",
                    territorial_indicators=[elem.attrib['territorial']],
                    legal_significance="Explicit territorial attribute",
                    xml_path=self._get_xpath(elem)
                ))
        
        return evidence
    
    def _analyze_extent_elements(self, root: ET.Element) -> List[TerritorialEvidence]:
        """Analyze territorial extent elements specifically"""
        evidence = []
        
        # Find all elements that might contain territorial extent information
        extent_elements = [
            ".//akn:territorialExtent",
            ".//akn:extent", 
            ".//akn:application",
            ".//akn:scope"
        ]
        
        for xpath in extent_elements:
            for elem in root.findall(xpath, self.namespace):
                if elem.text and elem.text.strip():
                    territories = self._extract_territories_from_text(elem.text)
                    if territories:
                        evidence.append(TerritorialEvidence(
                            xml_element=elem.tag.split('}')[-1],  # Remove namespace
                            text_snippet=elem.text.strip(),
                            pattern_type="territorial_extent",
                            confidence=0.9,
                            method="xml_structure",
                            territorial_indicators=territories,
                            legal_significance="Formal territorial extent in LegalDocML",
                            xml_path=self._get_xpath(elem)
                        ))
        
        return evidence
    
    def _analyze_section_content(self, root: ET.Element, section_id: Optional[str]) -> List[TerritorialEvidence]:
        """Analyze content within sections for territorial patterns"""
        evidence = []
        
        # Target specific section if provided
        if section_id:
            sections = root.findall(f".//akn:section[@eId='{section_id}']", self.namespace)
        else:
            sections = root.findall(".//akn:section", self.namespace)
        
        for section in sections:
            section_text = self._extract_text(section)
            section_id_attr = section.attrib.get('eId', 'unknown')
            
            # Apply all pattern categories to section text
            section_evidence = []
            
            # Check jurisdictional patterns
            section_evidence.extend(self._check_patterns(
                section_text, 
                self.jurisdictional_patterns, 
                "jurisdictional_authority",
                section_id_attr,
                section
            ))
            
            # Check federal-devolved patterns
            section_evidence.extend(self._check_patterns(
                section_text,
                self.federal_devolved_patterns,
                "federal_devolved_power", 
                section_id_attr,
                section
            ))
            
            # Check enforcement patterns
            section_evidence.extend(self._check_patterns(
                section_text,
                self.enforcement_patterns,
                "enforcement_variation",
                section_id_attr, 
                section
            ))
            
            # Check scope ambiguity patterns
            section_evidence.extend(self._check_patterns(
                section_text,
                self.scope_ambiguity_patterns,
                "territorial_scope_ambiguity",
                section_id_attr,
                section
            ))
            
            # Check constitutional patterns  
            section_evidence.extend(self._check_patterns(
                section_text,
                self.constitutional_patterns,
                "constitutional_competence",
                section_id_attr,
                section
            ))
            
            evidence.extend(section_evidence)
        
        return evidence
    
    def _apply_nlp_analysis(self, root: ET.Element, section_id: Optional[str]) -> List[TerritorialEvidence]:
        """Apply NLP analysis to extract territorial entities and relationships"""
        evidence = []
        
        if not self.nlp:
            return evidence
        
        # Extract all text content
        full_text = self._extract_text(root)
        
        # Process with spaCy
        doc = self.nlp(full_text[:1000000])  # Limit for performance
        
        # Extract territorial entities
        territorial_entities = []
        for ent in doc.ents:
            if ent.label_ in ['TERRITORY', 'AUTHORITY', 'COURT', 'GPE', 'ORG']:
                territorial_entities.append((ent.text, ent.label_, ent.start_char, ent.end_char))
        
        # Group nearby territorial entities (potential conflicts)
        entity_clusters = self._cluster_nearby_entities(territorial_entities, doc)
        
        for cluster in entity_clusters:
            if len(cluster) > 1:  # Multiple territorial entities near each other
                cluster_text = " ".join([ent[0] for ent in cluster])
                evidence.append(TerritorialEvidence(
                    xml_element="content",
                    text_snippet=cluster_text,
                    pattern_type="territorial_entity_cluster",
                    confidence=0.7,
                    method="nlp_entity",
                    territorial_indicators=[ent[0] for ent in cluster if ent[1] == 'TERRITORY'],
                    legal_significance="Multiple territorial entities in proximity suggest disambiguation issue"
                ))
        
        # Semantic similarity analysis
        semantic_evidence = self._semantic_territorial_analysis(full_text)
        evidence.extend(semantic_evidence)
        
        return evidence
    
    def _check_patterns(self, text: str, pattern_dict: Dict, category: str, section_id: str, xml_element: ET.Element) -> List[TerritorialEvidence]:
        """Check text against pattern dictionary"""
        evidence = []
        
        for pattern_type, patterns in pattern_dict.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Extract context around match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_pattern_confidence(pattern, match.group(), context)
                    
                    evidence.append(TerritorialEvidence(
                        xml_element=f"section_{section_id}",
                        text_snippet=context,
                        pattern_type=f"{category}_{pattern_type}",
                        confidence=confidence,
                        method="rule_based",
                        territorial_indicators=self._extract_territories_from_text(context),
                        legal_significance=f"Pattern match indicating {category.replace('_', ' ')}",
                        xml_path=self._get_xpath(xml_element)
                    ))
        
        return evidence
    
    def _extract_territories_from_text(self, text: str) -> List[str]:
        """Extract territory names from text"""
        territories = []
        territory_patterns = {
            'England': r'\bEngland\b',
            'Wales': r'\bWales\b', 
            'Scotland': r'\bScotland\b',
            'Northern Ireland': r'\bNorthern\s+Ireland\b',
            'England and Wales': r'\bEngland\s+and\s+Wales\b',
            'United Kingdom': r'\b(?:United\s+Kingdom|UK)\b'
        }
        
        for territory, pattern in territory_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                territories.append(territory)
        
        return territories
    
    def _calculate_pattern_confidence(self, pattern: str, match: str, context: str) -> float:
        """Calculate confidence score for pattern match"""
        base_confidence = 0.6
        
        # Boost confidence for specific legal terms
        legal_terms = ['shall', 'may', 'must', 'court', 'authority', 'jurisdiction', 'enforcement']
        if any(term in context.lower() for term in legal_terms):
            base_confidence += 0.2
        
        # Boost confidence for explicit territorial markers
        territorial_markers = ['england', 'wales', 'scotland', 'northern ireland']
        if any(marker in context.lower() for marker in territorial_markers):
            base_confidence += 0.15
        
        # Boost confidence for formal legal language
        if re.search(r'\b(?:subsection|paragraph|section)\s*\(\d+\)', context):
            base_confidence += 0.1
        
        return min(base_confidence, 0.95)
    
    def _cluster_nearby_entities(self, entities: List[Tuple], doc) -> List[List[Tuple]]:
        """Cluster territorial entities that appear near each other"""
        clusters = []
        used_entities = set()
        
        for i, entity in enumerate(entities):
            if i in used_entities:
                continue
                
            cluster = [entity]
            used_entities.add(i)
            
            # Find nearby entities (within 100 characters)
            for j, other_entity in enumerate(entities[i+1:], i+1):
                if j in used_entities:
                    continue
                if abs(entity[2] - other_entity[2]) < 100:  # Within 100 chars
                    cluster.append(other_entity)
                    used_entities.add(j)
            
            if len(cluster) > 0:
                clusters.append(cluster)
        
        return clusters
    
    def _semantic_territorial_analysis(self, text: str) -> List[TerritorialEvidence]:
        """Use semantic similarity to detect territorial disambiguation"""
        evidence = []
        
        try:
            # Split text into sentences
            sentences = re.split(r'[.!?]', text)
            
            # Vectorize sentences
            sentence_vectors = self.territorial_vectorizer.transform(sentences)
            
            # Find sentences with high similarity to territorial patterns
            similarity_threshold = 0.3
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 20:  # Skip very short sentences
                    continue
                
                sentence_vector = sentence_vectors[i:i+1]
                
                # Calculate similarity to territorial corpus
                similarities = cosine_similarity(sentence_vector, self.territorial_vectorizer.transform([
                    "territorial jurisdiction enforcement",
                    "devolved powers reserved matters", 
                    "different courts different territories",
                    "constitutional competence boundaries"
                ]))
                
                max_similarity = np.max(similarities)
                if max_similarity > similarity_threshold:
                    evidence.append(TerritorialEvidence(
                        xml_element="semantic_analysis",
                        text_snippet=sentence.strip()[:200],
                        pattern_type="semantic_territorial",
                        confidence=float(max_similarity),
                        method="semantic_similarity",
                        territorial_indicators=self._extract_territories_from_text(sentence),
                        legal_significance="Semantic similarity to territorial legal patterns"
                    ))
        
        except Exception as e:
            logger.warning(f"Semantic analysis failed: {e}")
        
        return evidence
    
    def _classify_disambiguation_type(self, evidence: List[TerritorialEvidence]) -> Tuple[DisambiguationType, float]:
        """Classify the type of territorial disambiguation using academic framework"""
        
        # Count evidence by category
        category_scores = {
            DisambiguationType.JURISDICTIONAL_AUTHORITY: 0,
            DisambiguationType.FEDERAL_DEVOLVED_POWER: 0,
            DisambiguationType.ENFORCEMENT_VARIATION: 0,
            DisambiguationType.TERRITORIAL_SCOPE_AMBIGUITY: 0,
            DisambiguationType.CONSTITUTIONAL_COMPETENCE: 0
        }
        
        total_confidence = 0
        evidence_count = 0
        
        for ev in evidence:
            evidence_count += 1
            total_confidence += ev.confidence
            
            # Map pattern types to academic categories
            if any(keyword in ev.pattern_type for keyword in ['jurisdiction', 'authority', 'competent']):
                category_scores[DisambiguationType.JURISDICTIONAL_AUTHORITY] += ev.confidence
            elif any(keyword in ev.pattern_type for keyword in ['devolved', 'reserved', 'federal']):
                category_scores[DisambiguationType.FEDERAL_DEVOLVED_POWER] += ev.confidence
            elif any(keyword in ev.pattern_type for keyword in ['enforcement', 'court', 'proceedings', 'penalty']):
                category_scores[DisambiguationType.ENFORCEMENT_VARIATION] += ev.confidence
            elif any(keyword in ev.pattern_type for keyword in ['scope', 'extent', 'application', 'ambiguous']):
                category_scores[DisambiguationType.TERRITORIAL_SCOPE_AMBIGUITY] += ev.confidence
            elif any(keyword in ev.pattern_type for keyword in ['constitutional', 'consent', 'competence']):
                category_scores[DisambiguationType.CONSTITUTIONAL_COMPETENCE] += ev.confidence
        
        if evidence_count == 0:
            return DisambiguationType.NO_DISAMBIGUATION, 0.0
        
        # Find highest scoring category
        max_category = max(category_scores.items(), key=lambda x: x[1])
        
        if max_category[1] == 0:
            return DisambiguationType.NO_DISAMBIGUATION, 0.0
        
        # Calculate overall confidence
        overall_confidence = min(total_confidence / evidence_count, 0.95)
        
        return max_category[0], overall_confidence
    
    def _generate_recommendations(self, disambiguation_type: DisambiguationType, evidence: List[TerritorialEvidence]) -> List[str]:
        """Generate recommendations based on disambiguation type"""
        recommendations = []
        
        if disambiguation_type == DisambiguationType.JURISDICTIONAL_AUTHORITY:
            recommendations = [
                "Clarify which courts have jurisdiction in each territory",
                "Add explicit territorial jurisdiction clauses",
                "Specify competent authorities for each jurisdiction"
            ]
        elif disambiguation_type == DisambiguationType.FEDERAL_DEVOLVED_POWER:
            recommendations = [
                "Clearly state which powers are reserved vs devolved",
                "Add devolution impact statements",
                "Include Legislative Consent Motion requirements where needed"
            ]
        elif disambiguation_type == DisambiguationType.ENFORCEMENT_VARIATION:
            recommendations = [
                "Specify enforcement bodies for each territory",
                "Clarify different court procedures by jurisdiction",
                "Add appeal route information for each territory",
                "Include penalty variation explanations"
            ]
        elif disambiguation_type == DisambiguationType.TERRITORIAL_SCOPE_AMBIGUITY:
            recommendations = [
                "Add explicit territorial extent clauses",
                "Clarify which sections apply to which territories",
                "Include territorial exceptions and exclusions",
                "Add interpretative guidance for ambiguous provisions"
            ]
        elif disambiguation_type == DisambiguationType.CONSTITUTIONAL_COMPETENCE:
            recommendations = [
                "Include Legislative Consent Motion requirements",
                "Clarify constitutional boundaries and competences",
                "Add ultra vires protection clauses",
                "Specify Sewel Convention compliance"
            ]
        else:
            recommendations = ["No specific territorial disambiguation issues detected"]
        
        return recommendations
    
    def _suggest_xml_enhancements(self, disambiguation_type: DisambiguationType, evidence: List[TerritorialEvidence]) -> List[str]:
        """Suggest LegalDocML XML enhancements for better territorial disambiguation"""
        enhancements = []
        
        if disambiguation_type == DisambiguationType.JURISDICTIONAL_AUTHORITY:
            enhancements = [
                "<akn:jurisdiction territory='EW' court='magistrates-court' appeals-to='crown-court'/>",
                "<akn:jurisdiction territory='S' court='sheriff-court' appeals-to='sheriff-appeal-court'/>",
                "<akn:competentAuthority territory='EW'>Competition and Markets Authority</akn:competentAuthority>"
            ]
        elif disambiguation_type == DisambiguationType.FEDERAL_DEVOLVED_POWER:
            enhancements = [
                "<akn:territorialExtent><akn:territory ref='EW' status='applies'/><akn:territory ref='S' status='applies'/><akn:territory ref='NI' status='devolved-exclusion'/></akn:territorialExtent>",
                "<akn:devolutionNote territory='NI' alternativeLegislation='Employment Act (Northern Ireland) 2016'/>",
                "<akn:reservedMatter>Statutory sick pay remains reserved despite employment law devolution</akn:reservedMatter>"
            ]
        elif disambiguation_type == DisambiguationType.ENFORCEMENT_VARIATION:
            enhancements = [
                "<akn:enforcementBody territory='EW'>CMA and local weights and measures authorities</akn:enforcementBody>",
                "<akn:enforcementBody territory='S'>CMA and Scottish local authorities</akn:enforcementBody>",
                "<akn:procedureVariation territory='S' specialRequirement='scottish-ministers-consent'/>"
            ]
        elif disambiguation_type == DisambiguationType.TERRITORIAL_SCOPE_AMBIGUITY:
            enhancements = [
                "<akn:territorialScope confidence='0.8' explicit='false'/>",
                "<akn:scopeAmbiguity>Territorial application unclear - requires interpretation</akn:scopeAmbiguity>",
                "<akn:interpretativeGuidance>This provision applies to England and Wales only despite general Act scope</akn:interpretativeGuidance>"
            ]
        elif disambiguation_type == DisambiguationType.CONSTITUTIONAL_COMPETENCE:
            enhancements = [
                "<akn:legislativeConsentRequired territory='S' assembly='scottish-parliament'/>",
                "<akn:constitutionalBoundary type='ultra-vires-risk'>Potential constitutional competence conflict</akn:constitutionalBoundary>",
                "<akn:sewelConvention status='requires-consent'>Legislative Consent Motion required</akn:sewelConvention>"
            ]
        
        return enhancements
    
    def _get_academic_classification(self, disambiguation_type: DisambiguationType) -> str:
        """Get academic legal classification for the disambiguation type"""
        classifications = {
            DisambiguationType.JURISDICTIONAL_AUTHORITY: "Territorial vs Personal vs Subject Matter Jurisdiction (International Law)",
            DisambiguationType.FEDERAL_DEVOLVED_POWER: "Reserved vs Devolved vs Concurrent Powers (Federalism Literature)",
            DisambiguationType.ENFORCEMENT_VARIATION: "Administrative Authority and Procedural Differences (Administrative Law)",
            DisambiguationType.TERRITORIAL_SCOPE_AMBIGUITY: "Statutory Interpretation of Geographic Application (Legal Interpretation Theory)",
            DisambiguationType.CONSTITUTIONAL_COMPETENCE: "Ultra Vires and Legislative Consent (Constitutional Law)",
            DisambiguationType.NO_DISAMBIGUATION: "No Territorial Legal Issues Identified"
        }
        return classifications.get(disambiguation_type, "Unknown Classification")
    
    def _get_xpath(self, element: ET.Element) -> str:
        """Generate XPath for XML element (simplified)"""
        try:
            # Simple XPath generation - in practice would need more sophisticated approach
            tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            parent = element.getparent() if hasattr(element, 'getparent') else None
            if parent is not None:
                parent_tag = parent.tag.split('}')[-1] if '}' in parent.tag else parent.tag
                return f"//{parent_tag}/{tag_name}"
            return f"//{tag_name}"
        except:
            return "unknown_xpath"
    
    def generate_enhanced_analysis_report(self, result: DisambiguationResult) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "act_analyzed": result.act_url,
                "section_analyzed": result.section_analyzed,
                "analyzer_version": "1.0",
                "academic_framework": "Evidence-based legal territorial disambiguation"
            },
            
            "disambiguation_analysis": {
                "type": result.disambiguation_type.value,
                "academic_classification": result.academic_classification,
                "confidence": result.confidence,
                "severity": self._assess_severity(result)
            },
            
            "territorial_findings": {
                "territories_identified": result.territories_identified,
                "legal_authorities": result.legal_authorities,
                "xml_elements_flagged": result.xml_elements_flagged,
                "evidence_count": len(result.evidence)
            },
            
            "detailed_evidence": [
                {
                    "xml_element": ev.xml_element,
                    "pattern_type": ev.pattern_type,
                    "confidence": ev.confidence,
                    "method": ev.method,
                    "text_snippet": ev.text_snippet[:200] + "..." if len(ev.text_snippet) > 200 else ev.text_snippet,
                    "legal_significance": ev.legal_significance,
                    "territorial_indicators": ev.territorial_indicators
                }
                for ev in result.evidence
            ],
            
            "recommendations": {
                "immediate_actions": result.recommendations,
                "xml_enhancements": result.suggested_xml_enhancements,
                "legal_review_required": self._assess_legal_review_need(result)
            },
            
            "academic_grounding": {
                "literature_basis": self._get_literature_basis(result.disambiguation_type),
                "legal_precedents": self._get_relevant_precedents(result.disambiguation_type),
                "comparative_jurisdictions": self._get_comparative_examples(result.disambiguation_type)
            }
        }
        
        return report
    
    def _assess_severity(self, result: DisambiguationResult) -> str:
        """Assess severity of territorial disambiguation issue"""
        if result.disambiguation_type == DisambiguationType.CONSTITUTIONAL_COMPETENCE:
            return "HIGH - Constitutional issues requiring legal review"
        elif result.disambiguation_type == DisambiguationType.FEDERAL_DEVOLVED_POWER:
            return "HIGH - Devolution conflicts with legal implications"
        elif result.disambiguation_type == DisambiguationType.JURISDICTIONAL_AUTHORITY:
            return "MEDIUM - Jurisdictional clarity needed"
        elif result.disambiguation_type == DisambiguationType.ENFORCEMENT_VARIATION:
            return "MEDIUM - Procedural differences across territories"
        elif result.disambiguation_type == DisambiguationType.TERRITORIAL_SCOPE_AMBIGUITY:
            return "LOW-MEDIUM - Interpretative guidance helpful"
        else:
            return "LOW - No significant territorial issues"
    
    def _assess_legal_review_need(self, result: DisambiguationResult) -> bool:
        """Determine if legal review is required"""
        high_risk_types = [
            DisambiguationType.CONSTITUTIONAL_COMPETENCE,
            DisambiguationType.FEDERAL_DEVOLVED_POWER
        ]
        return result.disambiguation_type in high_risk_types or result.confidence > 0.8
    
    def _get_literature_basis(self, disambiguation_type: DisambiguationType) -> List[str]:
        """Get academic literature basis for each category"""
        literature = {
            DisambiguationType.JURISDICTIONAL_AUTHORITY: [
                "Lotus Case (PCIJ 1927) - Territorial jurisdiction principles",
                "Oppenheim's International Law - Jurisdictional authority doctrine",
                "Cornell Law - Territorial vs personal jurisdiction"
            ],
            DisambiguationType.FEDERAL_DEVOLVED_POWER: [
                "Elazar's Federal Theory - Division of powers in federal systems",
                "Bogdanor's 'Devolution in the United Kingdom'",
                "Comparative federalism literature - reserved vs devolved powers"
            ],
            DisambiguationType.ENFORCEMENT_VARIATION: [
                "Wade & Forsyth 'Administrative Law' - Administrative authority principles",
                "Administrative law doctrine - Enforcement jurisdiction variations",
                "Multi-jurisdictional enforcement literature"
            ],
            DisambiguationType.TERRITORIAL_SCOPE_AMBIGUITY: [
                "Bennion on Statutory Interpretation - Territorial extent interpretation",
                "Legal drafting theory - Geographic application of statutes",
                "Statutory construction doctrine"
            ],
            DisambiguationType.CONSTITUTIONAL_COMPETENCE: [
                "Constitutional law theory - Ultra vires doctrine",
                "Sewel Convention literature - Legislative consent requirements",
                "Parliamentary sovereignty vs devolved competence"
            ]
        }
        return literature.get(disambiguation_type, ["General legal literature"])
    
    def _get_relevant_precedents(self, disambiguation_type: DisambiguationType) -> List[str]:
        """Get relevant legal precedents"""
        precedents = {
            DisambiguationType.JURISDICTIONAL_AUTHORITY: [
                "International Shoe v. Washington (territorial jurisdiction)",
                "World-Wide Volkswagen v. Woodson (jurisdictional limits)"
            ],
            DisambiguationType.FEDERAL_DEVOLVED_POWER: [
                "Reference re Secession of Quebec (federal-provincial relations)",
                "Scottish Parliament devolution cases"
            ],
            DisambiguationType.ENFORCEMENT_VARIATION: [
                "Administrative law cases on territorial enforcement",
                "Cross-border enforcement precedents"
            ],
            DisambiguationType.CONSTITUTIONAL_COMPETENCE: [
                "Miller cases (Brexit and constitutional competence)",
                "Legislative Consent Motion precedents"
            ]
        }
        return precedents.get(disambiguation_type, ["No specific precedents identified"])
    
    def _get_comparative_examples(self, disambiguation_type: DisambiguationType) -> List[str]:
        """Get comparative jurisdiction examples"""
        examples = {
            DisambiguationType.JURISDICTIONAL_AUTHORITY: [
                "US federal-state court jurisdiction",
                "Canadian federal-provincial court systems",
                "Australian state-federal jurisdiction"
            ],
            DisambiguationType.FEDERAL_DEVOLVED_POWER: [
                "Canadian federal-provincial division",
                "German federal-Länder distribution",
                "Australian Commonwealth-state powers"
            ],
            DisambiguationType.ENFORCEMENT_VARIATION: [
                "US federal agencies vs state enforcement",
                "EU enforcement variation across member states",
                "Canadian enforcement coordination"
            ]
        }
        return examples.get(disambiguation_type, ["No comparative examples identified"])

def main():
    """Main execution function with examples"""
    print("LegalDocML-Based Territorial Disambiguation Analyzer")
    print("=" * 60)
    
    # Example URLs to test
    test_urls = [
        "https://www.legislation.gov.uk/ukpga/2024/13",  # Digital Markets Act 2024
        "https://www.legislation.gov.uk/ukpga/1990/43",  # Environmental Protection Act 1990
    ]
    
    for url in test_urls:
        print(f"\nAnalyzing: {url}")
        print("-" * 40)
        
        try:
            # Initialize analyzer with LegalDocML
            analyzer = LegalDocMLTerritorialAnalyzer(url, is_section=False)
            
            # Analyze territorial disambiguation
            result = analyzer.analyze_territorial_disambiguation()
            
            # Generate comprehensive report
            report = analyzer.generate_enhanced_analysis_report(result)
            
            # Display key findings
            print(f"Disambiguation Type: {result.disambiguation_type.value}")
            print(f"Academic Classification: {result.academic_classification}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Severity: {report['disambiguation_analysis']['severity']}")
            
            print(f"\nTerritories Identified: {', '.join(result.territories_identified)}")
            print(f"Legal Authorities: {', '.join(result.legal_authorities[:3])}...")
            
            print(f"\nEvidence Found ({len(result.evidence)} items):")
            for i, evidence in enumerate(result.evidence[:3], 1):
                print(f"  {i}. {evidence.pattern_type} (confidence: {evidence.confidence:.2f})")
                print(f"     {evidence.text_snippet[:100]}...")
            
            print(f"\nRecommendations:")
            for rec in result.recommendations[:3]:
                print(f"  • {rec}")
            
            print(f"\nSuggested XML Enhancements:")
            for enhancement in result.suggested_xml_enhancements[:2]:
                print(f"  • {enhancement}")
            
            # Save detailed report
            report_filename = f"territorial_analysis_{url.split('/')[-1]}.json"
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nDetailed report saved: {report_filename}")
            
        except Exception as e:
            print(f"Error analyzing {url}: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check generated JSON reports for full details.")

if __name__ == "__main__":
    main()