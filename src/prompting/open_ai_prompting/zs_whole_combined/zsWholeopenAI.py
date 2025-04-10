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
    Extract comprehensive medical information from provided text, covering all major systems and historical data, and organize it into a single structured JSON object.
    """
    text = InputField(
        desc="medical text"
    )

    extract_info = OutputField(
        desc="""organize all extracted information into a single JSON output structure, consolidating data from all relevant systems, diagnostics, and history.
        Ensure all entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
        Focus on diagnostic clarity and exclude treatment details, figures, or references to tables.
        All entries should be explicitly tied to their respective systems or observations.""",
         prefix="""
            Extract information from the provided text and organize it into a JSON object.

            **Schema Rules:**
            1. Use only the following top-level keys and their nested structures:
            {
            "iem": { "is_IEM": list of strings },
            "Vitals_Hema": {
                  "temperature": list of strings,
                  "pulse": list of strings,
                  "respiratory_rate": list of strings,
                  "blood_pressure": list of strings,
                  "oxygen_saturation": list of strings,
                  "hematological_conditions": list of strings,
                  "hematology_lab_tests_measurements": list of strings
            },
            "Pregnancy": {
                  "neonatal_health": list of strings,
                  "maternal_health": list of strings,
                  "pregnancy_lab_tests_imaging_exam": list of strings
            },

            "Neuro": {
                  "neurological": list of strings,
                  "cognitive": list of strings,
                  "neuro_lab_tests_imaging_exam": list of strings
            },
            "EENT": {
                  "eyes": list of strings,
                  "ears": list of strings,
                  "nose": list of strings,
                  "throat": list of strings,
                  "EENT_lab_tests_imaging_exam": list of strings
            },
            "CVS": {
                  "cardiac": list of strings,
                  "vascular": list of strings,
                  "CVS_lab_tests_imaging_exam": list of strings
            },
            "RESP": {
                  "respiratory": list of strings,
                  "respiratory_system_lab_tests_imaging_exam": list of strings
            },
            "GI": {
                 "gastrointestinal": list of strings,
                 "gastrointestinal_lab_tests_imaging_exam": list of strings
            },
           "GU": {
                 "genital": list of strings,
                 "urinary": list of strings,
                 "GU_lab_tests_imaging_exam": list of strings

           },
          "DERM": {
                "skin_conditions": list of strings,
                "facial_features": list of strings,
                "breast_conditions": list of strings,
                "derm_breast_facial_lab_tests_image_exam": list of strings
           },
         "MSK": {
                "muscle": list of strings,
                "skeletal": list of strings,
                "MSK_lab_tests_image_exam": list of strings
         },
        "LYMPH": {
                "adenoid": list of strings,
                "tonsils": list of strings,
                "lymphatic_tissues": list of strings,
                "lymph_nodes": list of strings,
                "bone_marrow": list of strings,
                "spleen": list of strings,
                "immune_cells": list of strings,
                "Lymphatic_lab_tests_image_exam": list of strings
          },
        "ENDO": {
                "endocrine_glands": list of strings,
                "Endocrine_lab_tests_image_exam": list of strings
         },
            "History": {
                  "past_medical_history": list of strings,
                  "past_surgical_history": list of strings,
                  "history_of_present_illness": list of strings,
                  "social_history": list of strings,
                  "family_and_genetics_history": list of strings,
                  "chief_complaint": list of strings
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