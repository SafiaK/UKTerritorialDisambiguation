import json
import rdflib
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple
import re

# Define namespaces
PARL = Namespace("http://parliament.uk/ontologies/geographic-area/")
AKN = Namespace("http://docs.oasis-open.org/legaldocml/ns/akn/3.0/")
TLC = Namespace("http://docs.oasis-open.org/legaldocml/ns/akn/3.0/tlc/")
LEG = Namespace("https://www.legislation.gov.uk/def/legislation/")
DC = Namespace("http://purl.org/dc/elements/1.1/")
DCTERMS = Namespace("http://purl.org/dc/terms/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

class TerritorialOntologyGenerator:
    """Generate RDF/OWL ontology for UK territorial legal variations"""
    
    def __init__(self):
        self.graph = Graph()
        self._bind_namespaces()
        self._define_core_classes()
        self._define_properties()
        self._create_base_territories()
        
    def _bind_namespaces(self):
        """Bind all required namespaces"""
        self.graph.bind("parl", PARL)
        self.graph.bind("akn", AKN)
        self.graph.bind("tlc", TLC)
        self.graph.bind("leg", LEG)
        self.graph.bind("dc", DC)
        self.graph.bind("dcterms", DCTERMS)
        self.graph.bind("foaf", FOAF)
        self.graph.bind("geo", GEO)
        self.graph.bind("owl", OWL)
        self.graph.bind("skos", SKOS)
        
    def _define_core_classes(self):
        """Define core ontology classes following UK Parliament pattern"""
        # Geographic Area Classes
        self.graph.add((PARL.GeographicArea, RDF.type, OWL.Class))
        self.graph.add((PARL.GeographicArea, RDFS.label, Literal("Geographic Area")))
        
        self.graph.add((PARL.Country, RDF.type, OWL.Class))
        self.graph.add((PARL.Country, RDFS.subClassOf, PARL.GeographicArea))
        
        self.graph.add((PARL.JurisdictionalArea, RDF.type, OWL.Class))
        self.graph.add((PARL.JurisdictionalArea, RDFS.subClassOf, PARL.GeographicArea))
        
        # Legal Document Classes (extending with Akoma Ntoso concepts)
        self.graph.add((LEG.LegislativeDocument, RDF.type, OWL.Class))
        self.graph.add((LEG.LegislativeDocument, RDFS.subClassOf, AKN.Act))
        
        self.graph.add((LEG.LegislativeSection, RDF.type, OWL.Class))
        self.graph.add((LEG.LegislativeSection, RDFS.subClassOf, AKN.Section))
        
        # Territorial Application Classes
        self.graph.add((LEG.TerritorialApplication, RDF.type, OWL.Class))
        self.graph.add((LEG.TerritorialVariation, RDF.type, OWL.Class))
        
    def _define_properties(self):
        """Define ontology properties"""
        # Geographic properties
        self.graph.add((PARL.geometry, RDF.type, OWL.DatatypeProperty))
        self.graph.add((PARL.geometry, RDFS.domain, PARL.GeographicArea))
        
        # Territorial scope properties
        self.graph.add((LEG.territorialExtent, RDF.type, OWL.ObjectProperty))
        self.graph.add((LEG.territorialExtent, RDFS.domain, LEG.LegislativeDocument))
        self.graph.add((LEG.territorialExtent, RDFS.range, PARL.GeographicArea))
        
        self.graph.add((LEG.appliesTo, RDF.type, OWL.ObjectProperty))
        self.graph.add((LEG.appliesTo, RDFS.domain, LEG.LegislativeSection))
        self.graph.add((LEG.appliesTo, RDFS.range, PARL.JurisdictionalArea))
        
        # Variation properties
        self.graph.add((LEG.hasVariation, RDF.type, OWL.ObjectProperty))
        self.graph.add((LEG.hasVariation, RDFS.domain, LEG.LegislativeSection))
        self.graph.add((LEG.hasVariation, RDFS.range, LEG.TerritorialVariation))
        
        self.graph.add((LEG.variationText, RDF.type, OWL.DatatypeProperty))
        self.graph.add((LEG.variationText, RDFS.domain, LEG.TerritorialVariation))
        
    def _create_base_territories(self):
        """Create base UK territorial entities"""
        territories = {
            "england": ("England", "E92000001"),
            "wales": ("Wales", "W92000004"),
            "scotland": ("Scotland", "S92000003"),
            "northern_ireland": ("Northern Ireland", "N92000002"),
            "england_wales": ("England and Wales", "E+W"),
            "united_kingdom": ("United Kingdom", "UK")
        }
        
        for key, (label, code) in territories.items():
            uri = LEG[key]
            self.graph.add((uri, RDF.type, PARL.JurisdictionalArea))
            self.graph.add((uri, RDFS.label, Literal(label)))
            self.graph.add((uri, DCTERMS.identifier, Literal(code)))
            self.graph.add((uri, SKOS.notation, Literal(code)))
            
        # Add relationships
        self.graph.add((LEG.england_wales, PARL.contains, LEG.england))
        self.graph.add((LEG.england_wales, PARL.contains, LEG.wales))
        self.graph.add((LEG.united_kingdom, PARL.contains, LEG.england))
        self.graph.add((LEG.united_kingdom, PARL.contains, LEG.wales))
        self.graph.add((LEG.united_kingdom, PARL.contains, LEG.scotland))
        self.graph.add((LEG.united_kingdom, PARL.contains, LEG.northern_ireland))
        
    def process_territorial_json(self, json_data: Dict[str, Any], 
                               act_uri: str = None) -> Dict[str, Any]:
        """Process territorial disambiguation JSON and add to ontology"""
        results = {
            "sections_processed": 0,
            "variations_found": 0,
            "territorial_patterns": {},
            "structured_output": {}
        }
        
        if not act_uri:
            act_uri = "https://www.legislation.gov.uk/ukpga/2024/13"
            
        act_ref = URIRef(act_uri)
        self.graph.add((act_ref, RDF.type, LEG.LegislativeDocument))
        
        for section_id, territories in json_data.items():
            section_uri = URIRef(f"{act_uri}/{section_id}")
            self.graph.add((section_uri, RDF.type, LEG.LegislativeSection))
            self.graph.add((section_uri, DCTERMS.isPartOf, act_ref))
            self.graph.add((section_uri, DCTERMS.identifier, Literal(section_id)))
            
            # Process each territorial variation
            section_variations = {}
            for territory, text in territories.items():
                territory_key = self._normalize_territory_key(territory)
                
                if territory_key:
                    variation_uri = URIRef(f"{act_uri}/{section_id}/variation/{territory_key}")
                    self.graph.add((variation_uri, RDF.type, LEG.TerritorialVariation))
                    self.graph.add((section_uri, LEG.hasVariation, variation_uri))
                    
                    # Link to territorial area
                    territory_uri = self._get_territory_uri(territory_key)
                    if territory_uri:
                        self.graph.add((variation_uri, LEG.appliesTo, territory_uri))
                    
                    # Add variation text
                    self.graph.add((variation_uri, LEG.variationText, Literal(text)))
                    
                    # Add Akoma Ntoso metadata
                    self._add_akn_metadata(variation_uri, territory_key, text)
                    
                    section_variations[territory] = {
                        "normalized_territory": territory_key,
                        "text_snippet": text[:100] + "..." if len(text) > 100 else text,
                        "uri": str(variation_uri)
                    }
                    
                    results["variations_found"] += 1
                    
            results["structured_output"][section_id] = section_variations
            results["sections_processed"] += 1
            
        # Analyze territorial patterns
        results["territorial_patterns"] = self._analyze_patterns(json_data)
        
        return results
        
    def _normalize_territory_key(self, territory: str) -> str:
        """Normalize territory names to standard keys"""
        territory_lower = territory.lower().strip()
        
        mappings = {
            "england": "england",
            "wales": "wales",
            "scotland": "scotland",
            "northern ireland": "northern_ireland",
            "england and wales": "england_wales",
            "united kingdom": "united_kingdom",
            "uk": "united_kingdom"
        }
        
        return mappings.get(territory_lower, None)
        
    def _get_territory_uri(self, territory_key: str) -> URIRef:
        """Get URI for normalized territory key"""
        return LEG[territory_key] if territory_key else None
        
    def _add_akn_metadata(self, variation_uri: URIRef, territory: str, text: str):
        """Add Akoma Ntoso metadata for territorial scope"""
        # Add FRBRcountry equivalent
        country_code = self._get_country_code(territory)
        if country_code:
            self.graph.add((variation_uri, AKN.FRBRcountry, Literal(country_code)))
            
        # Add TLC location reference
        tlc_uri = TLC[f"location.{territory}"]
        self.graph.add((variation_uri, AKN.refersTo, tlc_uri))
        self.graph.add((tlc_uri, RDF.type, TLC.TLCLocation))
        self.graph.add((tlc_uri, RDFS.label, Literal(territory.replace("_", " ").title())))
        
    def _get_country_code(self, territory: str) -> str:
        """Get ISO country code for territory"""
        codes = {
            "england": "gb-eng",
            "wales": "gb-wls", 
            "scotland": "gb-sct",
            "northern_ireland": "gb-nir",
            "england_wales": "gb",
            "united_kingdom": "gb"
        }
        return codes.get(territory, "")
        
    def _analyze_patterns(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in territorial variations"""
        patterns = {
            "territorial_coverage": {},
            "common_patterns": [],
            "exclusive_territories": [],
            "inclusive_territories": []
        }
        
        # Count territorial occurrences
        territory_counts = {}
        for section, territories in json_data.items():
            for territory in territories.keys():
                normalized = self._normalize_territory_key(territory)
                if normalized:
                    territory_counts[normalized] = territory_counts.get(normalized, 0) + 1
                    
        patterns["territorial_coverage"] = territory_counts
        
        # Identify patterns
        for section, territories in json_data.items():
            territory_set = set(self._normalize_territory_key(t) for t in territories.keys() if self._normalize_territory_key(t))
            
            # Exclusive patterns (single territory)
            if len(territory_set) == 1:
                patterns["exclusive_territories"].append({
                    "section": section,
                    "territory": list(territory_set)[0]
                })
                
            # UK-wide patterns
            if territory_set == {"england", "wales", "scotland", "northern_ireland"}:
                patterns["common_patterns"].append({
                    "section": section,
                    "pattern": "uk_wide"
                })
                
        return patterns
        
    def export_ontology(self, format: str = "turtle") -> str:
        """Export the ontology in specified format"""
        return self.graph.serialize(format=format)
        
    def generate_structured_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured JSON with explicit territorial keys"""
        structured = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "source": "UK Territorial Legal Variations Ontology Generator",
                "akn_namespace": str(AKN),
                "parl_namespace": str(PARL)
            },
            "territories": {
                "england": {"code": "E92000001", "label": "England"},
                "wales": {"code": "W92000004", "label": "Wales"},
                "scotland": {"code": "S92000003", "label": "Scotland"},
                "northern_ireland": {"code": "N92000002", "label": "Northern Ireland"},
                "england_wales": {"code": "E+W", "label": "England and Wales"},
                "united_kingdom": {"code": "UK", "label": "United Kingdom"}
            },
            "sections": {}
        }
        
        for section_id, territories in json_data.items():
            section_data = {
                "section_id": section_id,
                "territorial_variations": {}
            }
            
            # Group by normalized territory
            for territory, text in territories.items():
                normalized = self._normalize_territory_key(territory)
                if normalized:
                    if normalized not in section_data["territorial_variations"]:
                        section_data["territorial_variations"][normalized] = {
                            "original_labels": [],
                            "text_variations": []
                        }
                    
                    section_data["territorial_variations"][normalized]["original_labels"].append(territory)
                    section_data["territorial_variations"][normalized]["text_variations"].append({
                        "text": text,
                        "length": len(text),
                        "hash": hash(text)
                    })
                    
            structured["sections"][section_id] = section_data
            
        return structured

def main():
    """Example usage"""
    # Sample JSON data (from your territorial_disambiguation_cases.json)
    
    # Read the sample data from the JSON file
    with open("education_act_territorial_disambiguation_cases.json", "r", encoding="utf-8") as f:
        sample_data = json.load(f)
  
    
    # Create generator
    generator = TerritorialOntologyGenerator()
    
    # Process JSON
    results = generator.process_territorial_json(sample_data)
    
    # Generate structured JSON
    structured = generator.generate_structured_json(sample_data)
    
    # Export ontology
    turtle_output = generator.export_ontology("turtle")
    
    # Print results
    # Save results to JSON files
    with open("processing_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open("structured_output.json", "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    print("Processing results saved to 'processing_results.json'")
    print("Structured JSON output saved to 'structured_output.json'")

if __name__ == "__main__":
    main()