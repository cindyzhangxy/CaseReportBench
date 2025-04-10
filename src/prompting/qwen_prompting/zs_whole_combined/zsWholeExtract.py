import dspy
import torch
import re
from dspy import (
    InputField,
    OutputField,
    Signature,
    Predict

)
import ast
import json
import os


os.environ['OLLAMA_NUM_PARALLEL'] = '28'


lm = dspy.OllamaLocal(model='qwen2:7b-instruct', max_tokens=2000, temperature=0.0, num_ctx=6000)
dspy.settings.configure(lm=lm)


# In[33]:


def str_2_dict(input_string, retry_count=0):
    try:
        # Extract JSON substring from the input string
        start_index = input_string.index('{')
        end_index = input_string.rindex('}') + 1
        dict_string = input_string[start_index:end_index]

        # Clean and convert the dictionary string to a dictionary
        dict_string = dict_string.replace('\n', '').replace("'", '"')
        return json.loads(dict_string)

    except (ValueError, SyntaxError) as e:
        print(f"Error processing input string: {str(e)}. Attempting to correct...")
        return error_correction(input_string, retry_count)  # Call the error correction function
    except Exception as e:
        print(f"Unexpected error: {str(e)}. Attempting to correct...")
        return error_correction(input_string, retry_count)  # Call the error correction function for any exception



class Correction(Signature):
    str_input = InputField(desc="Input text containing an error message along with an erroneous JSON structure.")
    corrected_output = OutputField(
        desc="Output a corrected, JSON-compliant dictionary. If no JSON-like substring is present, output YAML format.",
        prefix="""This task involves analyzing the provided text to identify any error messages and the associated erroneous JSON structure.
                  The goal is to correct the JSON formatting errors indicated by the message. Adjust keys, values, and the overall
                  format to ensure the output is a valid JSON-compliant dictionary. Corrections should strictly adhere to the issues
                  highlighted in the error message, without making assumptions beyond the provided text."""
    )


def error_correction(mal_formatted_json, retry_count=1):
    if retry_count > 2:  # Limit the retries to avoid infinite loops
        # print("Maximum retry attempts reached. Returning last corrected output.")
        return mal_formatted_json

    # Simulate a prediction for correction (Assuming Predict and Correction are defined)
    subject_pred = Predict(Correction)
    prediction = subject_pred(str_input=mal_formatted_json)
    extracted_dict = str_2_dict(prediction.corrected_output, retry_count + 1)

    if extracted_dict is None:
        # print("No valid JSON could be extracted after correction attempt.")
        return prediction.corrected_output  # Return the last attempted correction
    else:
        return extracted_dict


def simple_extraction(text, subject_heading):
    """Extract str as a dictionary based on the subject headings e.g. Demographics, history"""
    try:
        subject_pred = Predict(subject_heading)
        prediction = subject_pred(text= text)
        extracted_dict = str_2_dict(prediction.extract_info)
        if extracted_dict is not None:
            # print(prediction.extract_info)
            return extracted_dict
        else:
            raise ValueError("No dictionary could be extracted.")  # Force the function to handle as if an error occurred
    except Exception as e:
        # print(f"Error during extraction or conversion: {e}")
        # print("Returning raw extract info due to error in conversion.")
        return prediction


class SYSTEMIC_EXTRACTION(Signature):
    """
    Extract comprehensive medical information from provided text, covering all major systems and historical data, and organize it into a single structured JSON object.
    This few-shot class uses examples to demonstrate the expected format and guides extraction of observations, diagnostic results, imaging findings, and historical data relevant to the patient’s condition while excluding treatment details and unrelated systemic conditions.
    """
    text = InputField(
        desc="Comprehensive medical text, including observations, diagnostic results, history, and findings across all systems."
    )

    extract_info = OutputField(
        desc="""organize all extracted information into a single JSON-compliant object, consolidating data from all relevant systems, diagnostics, and history.
        Ensure all entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
        Focus on diagnostic clarity and exclude treatment details, figures, or references to tables.
        All entries should be explicitly tied to their respective systems or observations.""",
        prefix="""
            Extract information from the provided text and organize it into a JSON object.

            **Schema Rules:**
            1. Use only the following top-level keys and their nested structures:
            {
            "IEM": { "is_IEM": boolean },
            "pregnancy": {
                  "neonatal_health": list,
                  "maternal_health": list,
                  "pregnancy_lab_tests_imaging_exam": list of lab test or imaging relating to pregnancy or neonate 
            },
            "Vitals_Hema": {
                  "temperature": list,
                  "pulse": list,
                  "respiratory_rate": list,
                  "blood_pressure": list,
                  "oxygen_saturation": list,
                  "hematological_conditions": list,
                  "hematology_lab_tests_measurements": list of lab test or imaging relating to hematology and/or vitals
            },
            "Neuro": {
                  "neurological": list,
                  "cognitive": list,
                  "neuro_lab_tests_imaging_exam": list of lab test or imaging relating to neurology and cognition
            },
 	    "EENT": {
                  "eyes": list,
                  "ears": list,
		  "nose": list,
		  "throat": list,
                  "EENT_lab_tests_imaging_exam": list of lab test or imaging relating to ears, eyes, nose and throat
            },
	    "CVS": {
                  "cardiac": list,
                  "vascular": list,
                  "CVS_lab_tests_imaging_exam": list of lab test or imaging relating to cardiovascular system
            },
	    "RESP": {
                  "respiratory": list,
                  "respiratory_system_lab_tests_imaging_exam": list of lab test or imaging relating to respiratory system
            },
	    "GI": {
		 "genital": list,
	 	 "urinary": list,
		 "GU_lab_tests_imaging_exam":list of lab test or imaging relating to genital or urinary system
		
	   }, 
	   "GU": {
		 "gastrointestinal": list,
	 	 "gastrointestinal_lab_tests_imaging_exam: list of lab test or imaging relating to gastrointestinal system
		
	   }, 
	  "DERM": {
		"skin_conditions": list,
		"facial_features": list,
		"breast_conditions": list,
		"derm_breast_facial_lab_tests_image_exam: list of lab test or imaging relating to dermatology, facial and/or breast 
	   }, 
	 "MSK":{
		"muscle": list,
		"skeletal": list,
		"MSK_lab_tests_image_exam" : list of lab test or imaging relating to musculoskeletal system,
	"LYMPH": {
		'adenoid': List,
                'tonsils': List, 
                'lymphatic_tissues': List, 
                'lymph_nodes': List,
                'bone_marrow': List,
                'spleen': List,
                'immune_cells': List, 
                'Lymphatic_lab_tests_image_exam': list of lab test or imaging relating to musculoskeletal system,

	  },
	"ENDO": {
		'endocrine_glands': list,
		'Endocrine_lab_tests_image_exam': list of lab test or imaging relating to endocrine system
	 },
            "History": {
                  "past_medical_history": list,
                  "past_surgical_history": list,
                  "history_of_present_illness": string,
                  "social_history": list,
                  "family_and_genetics_history": list,
                  "chief_complaint": string
            }
            }

            2. Include only these keys and sub-keys. Do not add any extra keys or fields.
            3. Ensure the output JSON is valid and adheres strictly to this schema.
            4. For fields with no data, provide an empty list []

            **Output Requirements:**
            - The JSON object must be well-formed and strictly follow the schema.
            - Field values should be directly derived from the text.
            - Avoid assumptions; include only information explicitly present in the text.

"""
)
