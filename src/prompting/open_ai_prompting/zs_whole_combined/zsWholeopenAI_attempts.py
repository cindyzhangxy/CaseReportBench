#!/usr/bin/env python
# coding: utf-8

import re
import openai
import dspy
from dspy import (
    InputField,
    OutputField,
    Signature,
    Predict
)
import json
import os


api_key =  os.getenv('OPENAI_API_KEY')
model="gpt-4o"
lm = dspy.LM(model=model, temperature=0.0, max_tokens=2000, api_key=api_key)
dspy.configure(lm=lm)


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
    Extract comprehensive medical information from provided text, covering all major systems and historical data, 
    and organize it into a single structured JSON object. This few-shot class uses examples to demonstrate the expected 
    format and guides the extraction of observations, diagnostic results, imaging findings, and historical data 
    relevant to the patient’s condition, while excluding treatment details and unrelated systemic conditions.
    """

    text = InputField(
        desc="Medical text containing patient information"
    )

    extract_info = OutputField(
    desc = """Organize all extracted information into a single JSON-compliant object, consolidating data from all relevant systems, 
    diagnostics, and history. Ensure all entries are verbatim quotes or clear deductions based on the provided text, without making assumptions. 
    Focus on diagnostic clarity and exclude treatment details, figures, or references to tables. 
    Explicitly tie all entries to their respective systems or observations.""",

    prefix = """
    Extract the relevant information from the provided text into a JSON object with the following rules:
    
    **Structure Requirements**:
    - The JSON object must contain the following top-level keys and their respective nested structures:
      - "Pregnancy": {
            "neonatal_health": list,
            "maternal_health": list,
            "pregnancy_lab_tests_imaging_exam": list
        }
      - "Vitals_Hema": {
            "temperature": list,
            "pulse": list,
            "respiratory_rate": list,
            "blood_pressure": list,
            "oxygen_saturation": list,
            "hematological_conditions": list,
            "hematology_lab_tests_measurements": list
        }
      - "Neuro": {
            "neurological": list,
            "cognitive": list,
            "neuro_lab_tests_imaging_exam": list
        }
      - "EENT": {
            "eyes": list,
            "ears": list,
            "nose": list,
            "throat": list,
            "EENT_lab_tests_imaging_exam": list
        }
      - "CVS": {
            "cardiac": list,
            "vascular": list,
            "CVS_lab_tests_imaging_exam": list
        }
      - "RESP": {
            "respiratory": list,
            "respiratory_system_lab_tests_imaging_exam": list
        }
      - "GI": {
            "gastrointestinal": list,
            "gastrointestinal_lab_tests_imaging_exam": list
        }
      - "GU": {
            "genital": list,
            "urinary": list,
            "GU_lab_tests_imaging_exam": list
        }
      - "DERM": {
            "skin_conditions": list,
            "facial_features": list,
            "breast_conditions": list,
            "derm_breast_facial_lab_tests_image_exam": list
        }
      - "MSK": {
            "muscle": list,
            "skeletal": list,
            "MSK_lab_tests_image_exam": list
        }
      - "LYMPH": {
            "adenoid": list,
            "tonsils": list,
            "lymphatic_tissues": list,
            "lymph_nodes": list,
            "bone_marrow": list,
            "spleen": list,
            "immune_cells": list,
            "Lymphatic_lab_tests_image_exam": list
        }
      - "ENDO": {
            "endocrine_glands": list,
            "Endocrine_lab_tests_image_exam": list
        }
      - "History": {
            "past_medical_history": list,
            "past_surgical_history": list,
            "history_of_present_illness": list,
            "social_history": list,
            "family_and_genetics_history": list,
            "chief_complaint": list
        }
      - "Lab_Image": list (A consolidated list of lab tests and imaging findings across all systems)
    
    **Guidelines**:
    - Extract lab tests and imaging findings into their respective system keys and consolidate them into "Lab_Image."
    - Include all top-level keys in the JSON object, even if their values are empty lists (`[]`).
    - Avoid assumptions. Include only information explicitly stated in the text.
    - Derive entries verbatim from the text or through clear, evidence-based deductions.
    - Exclude treatment details, figures, or unrelated conditions.
"""
    )