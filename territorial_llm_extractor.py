import json
import os
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Initialize OpenAI client using API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in your environment or .env file.")
client = OpenAI(api_key=OPENAI_API_KEY)

def extract_territorial_differences(section_id: str, section_text: str) -> dict:
    """
    Calls OpenAI Chat Completion to extract territorial differences from a legal section text.
    Input: section_id (str), section_text (str)
    Output: Structured JSON dict with fields: section_id, full_text_reference, enforcement_variations, key_differences.
    Functionality:
      - Builds a prompt with a few-shot example and instructions.
      - Calls OpenAI's GPT-4 model to analyze the section and return a JSON summary.
      - Parses and returns the JSON output, or raises an error if the response is not valid JSON.
    """
    with open("data/2024/13/section-79.txt", "r", encoding="utf-8") as f:
        few_shot_example_input = f.read()
    few_shot_example_output = """
{
  "section_id": "section-79",
  "full_text_reference": "Subsections (1)–(7) and (11)–(14) are uniform across all territories; enforcement differences appear in (8)–(10).",
  "enforcement_variations": [
    {
      "territory": "England & Wales",
      "text": "(8) In England and Wales, such an amount is recoverable— (a) if the county court so orders, as if it were payable under an order of that court; (b) if the High Court so orders, as if it were payable under an order of that court."
    },
    {
      "territory": "Scotland",
      "text": "(9) In Scotland, such an amount may be enforced in the same manner as an extract registered decree arbitral bearing a warrant for execution issued by the sheriff court of any sheriffdom in Scotland."
    }
  ],
  "key_differences": [
    "England & Wales: recovery via court orders.",
    "Scotland: enforcement via registered arbitral decree."
  ]
}
"""
    # Build prompt
    prompt = f"""
You are a legal text analysis assistant.

You are given a section of UK legislation and you need to extract the territorial differences in the section.

Instructions:
1. Identify any parts of the text that apply to different UK territories (England, Wales, Scotland, Northern Ireland, or England & Wales).
2. Extract the exact text relevant to each territory separately.
3. Output the results in strict JSON format with the following fields:
   - section_id: string (use the provided section_id)
   - full_text_reference: short description indicating which parts are uniform and which vary
   - enforcement_variations: an array of objects, each with territory and text
   - key_differences: concise bullet points highlighting differences

Here is a few-shot example:

Input:
{few_shot_example_input}

Output:
{few_shot_example_output}

Now process the following section:

Section text:
{section_text}

Output Format:
{{
  "section_id": "section-XYZ",
  "full_text_reference": "...",
  "enforcement_variations": [...],
  "key_differences": [...]
}}
"""
    # Call the model
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful and precise legal text analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    # Extract text response
    response_text = (response.choices[0].message.content or "").strip()
    # Attempt to parse JSON
    try:
        output_json = json.loads(response_text)
    except json.JSONDecodeError:
        raise ValueError(f"Model response was not valid JSON:\n{response_text}")
    return output_json

def process_sections_in_one_go_second_method(sections: str,output_path:str) -> List[Dict[str, Any]]:
   prompt = f"""
You are a legal text analysis assistant.

You are given a list of sections of UK legislation. Your task is to evaluate every section for territorial differences.

After completing your analysis, you must return two outputs:
  - territorial_differences
  - other_sections

For each section or group of related sections:


To fins a substantive territorial difference
(- Review all the sections together, as they may define similar provisions differently in different territories (for example, one section applying to England and another applying to Wales).
- Identify any parts of the text that apply differently across UK territories (England, Wales, Scotland, Northern Ireland, or England & Wales).), 
output a structured object in the territorial_differences list.
- For each difference you identify, output a structured JSON object with these fields:
    - difference_id: a unique identifier string for this difference.
    - provisions: a dictionary mapping each section to the list of territories it applies to.
      Example:
      {{
        "section-47": ["England", "Wales"],
        "section-50": ["England"]
      }}
    - difference_type: categorize the difference. Choose only one of these labels:
        - "content" (the substantive legal rule or definition differs)
        - "scope" (the provision is explicitly applicable to some territories but not others)
        - "penalty" (different penalties or sanctions)
        - "procedure" (different procedural rules)
        - "enforcement_body" (different authority or body responsible for enforcement)
        - "other" (if none of the above apply)
    - what_is_different: a short label (e.g., "Definition of denominational education", "Territorial extent of duty").
    - why_is_different: a short explanation of why the difference exists (e.g., "devolved governance", "policy divergence", "historical separation of legal systems").
    - description: a precise explanation of what is different, citing the exact text that creates the difference. Be as specific as possible, and quote relevant sentences or phrases.
    - territory_texts: a dictionary mapping each territory to the exact excerpt of the text that applies to that territory.

If there is no substantive territorial difference, group the section(s) into other_sections by shared category and explanation.
Categories for sections with no difference:
- For each category you identify, output a structured JSON object with these fields:
    - type_id: a unique identifier string for this difference.
    - provisions: a dictionary mapping each section to the list of territories it applies to.
        Example:
        {{
          "section-47": ["England", "Wales"],
          "section-50": ["England"]
        }}
    - category: the category of the section

      - explicit_extent_clause: The section explicitly states where it applies but does not change the content.
      - uniform_provision_with_territorial_reference: The provision applies identically across all territories but mentions them to avoid ambiguity.
      - reserved_powers_devolution: The section references a territory only to note devolution or reserved powers without any substantive difference.
      - other: If none of the above applies (explain clearly).

Note:
- Each section must be evaluated for territorial differences and hence should be in either territorial_differences or other_sections.
- Make sure to pick up the territories of the section as they are in the given data for provisions field.
- territory_texts should be the exact text of the section that applies to the territory.
- Don't include any other text in the territory_texts field.
- Don't include any thing from your own knowledge or assumptions.
- Review all these instructions carefully before outputting the results.
Example:
{{
  "territorial_differences": [
    {{
      "difference_id": "diff_001",
      "provisions": {{
        "section-47": ["England", "Wales"]
      }},
      "difference_type": "content",
      "what_is_different": "Definition of 'denominational education'",
      "why_is_different": "Different statutory definitions in England and Wales",
      "description": "Section 47 defines denominational education in England as 'education provided in accordance with the tenets of a particular religion', while Section 50 defines it in Wales as 'education reflecting the traditions of a faith group'.",
      "territory_texts": {{
        "England": "In this Part  “ denominational education ”, in relation to a school in England , means religious education which— (a) is required by section 80(1)(a) or 101(1)(a) of the Education Act 2002 (c. 32) to be included in the school's basic curriculum, but (b) is not required by any enactment to be given in accordance with an agreed syllabus",
        "Wales": "In this Part, “ denominational education ”, in relation to a school in Wales, means teaching and learning in respect of Religion, Values and Ethics"
      }}
    }}
  ],
  "other_sections": [
    {{
      "provisions": {{
        "section-122": ["England", "Wales"]
      }},
      "category": "explicit_extent_clause",
      "description": "This section provides general definitions and clarifies that parts of the Act are to be read with the Education Act 1996. Its references to Wales do not create different obligations—only clarify jurisdiction."
    }}
  ]
}}

{sections}
"""
    # Call the model
   response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful and precise legal text analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    # Extract text response
   response_text = (response.choices[0].message.content or "").strip()
    # Attempt to parse JSON
   with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response_text, f, indent=2, ensure_ascii=False)
   return response_text

def process_sections_in_one_go(sections: str,output_path:str) -> List[Dict[str, Any]]:
   prompt = f"""
You are a legal text analysis assistant.

You are given multiple sections of UK legislation. Your task is to extract detailed territorial differences.

Instructions:

- Review all the sections together, as they may define similar provisions differently in different territories (for example, one section applying to England and another applying to Wales).
- Identify any parts of the text that apply differently across UK territories (England, Wales, Scotland, Northern Ireland, or England & Wales).
- For each difference you identify, output a structured JSON object with these fields:

  - difference_id: a unique identifier string for this difference.
  - provisions: a dictionary mapping each section to the list of territories it applies to.
    Example:
    {{
      "section-47": ["England", "Wales"],
      "section-50": ["England"]
    }}
  - difference_type: categorize the difference. Choose only one of these labels:
      - "content" (the substantive legal rule or definition differs)
      - "enforcement_body" (different institution or authority applies)
      - "scope" (the provision is explicitly applicable to some territories but not others)
      - "penalty" (different penalties or sanctions)
      - "procedure" (different procedural rules)
      - "other" (if none of the above apply)
  - what_is_different: a short label (e.g., "Definition of denominational education", "Territorial extent of duty").
  - why_is_different: a short explanation of why the difference exists (e.g., "devolved governance", "policy divergence", "historical separation of legal systems").
  - description: a precise explanation of what is different, citing the exact text that creates the difference. Be as specific as possible, and quote relevant sentences or phrases.
  - territory_texts: a dictionary mapping each territory to the exact excerpt of the text that applies to that territory.

Example:
[
  {{
    "difference_id": "diff_001",
    "provisions": {{
      "section-47": ["England", "Wales"]
    }},
    "difference_type": "content",
    "what_is_different": "Definition of 'denominational education'",
    "why_is_different": "Different statutory definitions in England and Wales",
    "description": "Section 47 defines denominational education in England as 'education provided in accordance with the tenets of a particular religion', while Section 50 defines it in Wales as 'education reflecting the traditions of a faith group'.",
    "territory_texts": {{
      "England": "In this Part  “ denominational education ”, in relation to a school in England , means religious education which— (a) is required by section 80(1)(a) or 101(1)(a) of the Education Act 2002 (c. 32) to be included in the school's basic curriculum, but (b) is not required by any enactment to be given in accordance with an agreed syllabus",
      "Wales": "In this Part, “ denominational education ”, in relation to a school in Wales, means teaching and learning in respect of Religion, Values and Ethics"
    }}
  }}
]

{sections}
"""
    # Call the model
   response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and precise legal text analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    # Extract text response
   response_text = (response.choices[0].message.content or "").strip()
    # Attempt to parse JSON
   with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response_text, f, indent=2, ensure_ascii=False)
   return response_text

def process_sections_llm(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes a list of sections (each with 'id' and 'text') using extract_territorial_differences.
    Input: List of dicts with 'id' and 'text'.
    Output: List of structured JSON results from the LLM for each section.
    Functionality:
      - Iterates over each section, calls extract_territorial_differences, and collects the results.
      - Returns a list of all LLM outputs for further aggregation or saving.
    """
    results = []
    for section in sections:
        print(f"Processing section {section['id']}...")
        try:
            result = extract_territorial_differences(section['id'], section['text'])
            results.append(result)
        except Exception as e:
            print(f"Error processing section {section['id']}: {e}")
    return results 

    