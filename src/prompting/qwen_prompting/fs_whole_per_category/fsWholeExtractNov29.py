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
lm = dspy.LM(model=model, max_tokens=2000, api_key=api_key)
dspy.configure(lm=lm)

# In[33]:


def str_2_dict(input_string):
    try:
        start_index = input_string.index('{')
        end_index = input_string.rindex('}') + 1
        dict_string = input_string[start_index:end_index]
        dict_string = dict_string.replace('\n', '').replace("'", '"')
        return json.loads(dict_string)
    except Exception as e:
        print(f"Error processing input string: {str(e)}. Returning placeholder {{}} and continuing...")
        return {}  # Return placeholder




class Correction(Signature):
    str_input = InputField(desc="Input text containing an error message along with an erroneous JSON structure.")
    corrected_output = OutputField(
        desc="Output a corrected, JSON-compliant dictionary. If no JSON-like substring is present, output YAML format.",
        prefix="""This task involves analyzing the provided text to identify any error messages and the associated erroneous JSON structure. 
                  The goal is to correct the JSON formatting errors indicated by the message. Adjust keys, values, and the overall 
                  format to ensure the output is a valid JSON-compliant dictionary. Corrections should strictly adhere to the issues 
                  highlighted in the error message, without making assumptions beyond the provided text."""
    )


def error_correction(mal_formatted_json, retry_count=0, max_retries=1):
    if retry_count >= max_retries:
        print("Maximum retry attempts reached. Skipping correction.")
        return {}  # Return placeholder after max retries

    subject_pred = Predict(Correction)
    prediction = subject_pred(str_input=mal_formatted_json)

    extracted_dict = str_2_dict(prediction.corrected_output)
    if extracted_dict == {}:  # If parsing fails, increment retry and retry
        return error_correction(prediction.corrected_output, retry_count + 1, max_retries)
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
        prefix="""Using the provided text, extract and organize the information into a comprehensive JSON-compliant dictionary. Use the examples below to guide the expected output format and structure:

                  Example 1:
                  Input:
                  "A 3-month-old infant presents with poor feeding, vomiting, and developmental delay. Lab results show elevated ammonia levels, metabolic acidosis, and hyperglycemia. The family history reveals a sibling with a similar condition who passed away at 2 years old. Genetic testing confirmed a mutation in the OTC gene."

                  Output:
                  {
                    "IEM": { "is_IEM": true },
                    "pregnancy": {
                      "neonatal_health": ["Poor feeding", "Vomiting", "Developmental delay"],
                      "maternal_health": [],
                      "pregnancy_tests_imaging_exam": ["Elevated ammonia levels", "Metabolic acidosis", "Hyperglycemia", "Mutation in OTC gene"]
                    },
                    "vitals_hematology": {
                      "temperature": [],
                      "pulse": [],
                      "respiratory_rate": [],
                      "blood_pressure": [],
                      "oxygen_saturation": [],
                      "hematological_conditions": [],
                      "hematology_tests_measurements": ["Elevated ammonia levels", "Metabolic acidosis", "Hyperglycemia"]
                    },
                    "immune": {
                      "immunity_conditions": [],
                      "immune_cell_counts": [],
                      "immunophenotyping": [],
                      "autoimmune_markers": [],
                      "inflammatory_markers": [],
                      "serology": [],
                      "infectious_disease_status": [],
                      "malignancy_biomarkers": [],
                      "malignancy_condition": [],
                      "immunology_exam_image": []
                    },
                    "neurology": {
                      "neurological": ["Developmental delay"],
                      "cognitive": [],
                      "neuro_tests_imaging_exam": []
                    },
                    "EENT": {
                      "eyes": [],
                      "ears": [],
                      "nose": [],
                      "throat": [],
                      "EENT_tests_image_exam": []
                    },
                    "CVS": {
                      "cardiac": [],
                      "vascular": [],
                      "CVS_tests_image_exam": []
                    },
                    "RESP": {
                      "respiratory": [],
                      "RESP_tests_image_exam": []
                    },
                    "GI": {
                      "gastrointestinal": [],
                      "GI_tests_image_exam": []
                    },
                    "GU": {
                      "urinary": [],
                      "genital": [],
                      "GU_tests_image_exam": []
                    },
                    "DERM": {
                      "skin_conditions": [],
                      "facial_features": [],
                      "breast_conditions": [],
                      "derm_breasts_facial_tests_image_exam": []
                    },
                    "MSK": {
                      "muscle": [],
                      "skeletal": [],
                      "MSK_tests_image_exam": []
                    },
                    "LYMPH": {
                      "adenoid": [],
                      "tonsils": [],
                      "lymphatic_tissues": [],
                      "lymph_nodes": [],
                      "thymus": [],
                      "bone_marrow": [],
                      "spleen": [],
                      "immune_cells": [],
                      "Lymphatic_tests_image_exam": []
                    },
                    "ENDO": {
                      "endocrine_glands": [],
                      "Endocrine_tests_image_exam": []
                    },
                    "history": {
                      "past_medical_history": [],
                      "past_surgical_history": [],
                      "history_of_present_illness": ["Poor feeding", "Vomiting", "Developmental delay"],
                      "social_history": [],
                      "family_and_genetics_history": ["Sibling with a similar condition who passed away at 2 years old"],
                      "chief_complaint": ["Poor feeding", "Vomiting"]
                    }
                  }

                  Example 2:
                  Input:
                  "A 45-year-old female presents with fatigue, cold intolerance, and weight gain. Physical exam reveals dry skin and periorbital edema. Labs show TSH of 15 mIU/L and low free T4, consistent with hypothyroidism."

                  Output:
                  {
                    "IEM": { "is_IEM": false },
                    "pregnancy": {
                      "neonatal_health": [],
                      "maternal_health": [],
                      "pregnancy_tests_imaging_exam": []
                    },
                    "vitals_hematology": {
                      "temperature": [],
                      "pulse": [],
                      "respiratory_rate": [],
                      "blood_pressure": [],
                      "oxygen_saturation": [],
                      "hematological_conditions": [],
                      "hematology_tests_measurements": []
                    },
                    "immune": {
                      "immunity_conditions": [],
                      "immune_cell_counts": [],
                      "immunophenotyping": [],
                      "autoimmune_markers": [],
                      "inflammatory_markers": [],
                      "serology": [],
                      "infectious_disease_status": [],
                      "malignancy_biomarkers": [],
                      "malignancy_condition": [],
                      "immunology_exam_image": []
                    },
                    "neurology": {
                      "neurological": [],
                      "cognitive": [],
                      "neuro_tests_imaging_exam": []
                    },
                    "EENT": {
                      "eyes": [],
                      "ears": [],
                      "nose": [],
                      "throat": [],
                      "EENT_tests_image_exam": []
                    },
                    "CVS": {
                      "cardiac": [],
                      "vascular": [],
                      "CVS_tests_image_exam": []
                    },
                    "RESP": {
                      "respiratory": [],
                      "RESP_tests_image_exam": []
                    },
                    "GI": {
                      "gastrointestinal": [],
                      "GI_tests_image_exam": []
                    },
                    "GU": {
                      "urinary": [],
                      "genital": [],
                      "GU_tests_image_exam": []
                    },
                    "DERM": {
                      "skin_conditions": ["Dry skin"],
                      "facial_features": ["Periorbital edema"],
                      "breast_conditions": [],
                      "derm_breasts_facial_tests_image_exam": []
                    },
                    "MSK": {
                      "muscle": [],
                      "skeletal": [],
                      "MSK_tests_image_exam": []
                    },
                    "LYMPH": {
                      "adenoid": [],
                      "tonsils": [],
                      "lymphatic_tissues": [],
                      "lymph_nodes": [],
                      "thymus": [],
                      "bone_marrow": [],
                      "spleen": [],
                      "immune_cells": [],
                      "Lymphatic_tests_image_exam": []
                    },
                    "ENDO": {
                      "endocrine_glands": ["Hypothyroidism"],
                      "Endocrine_tests_image_exam": ["TSH of 15 mIU/L", "Low free T4"]
                    },
                    "history": {
                      "past_medical_history": [],
                      "past_surgical_history": [],
                      "history_of_present_illness": ["Fatigue", "Cold intolerance", "Weight gain"],
                      "social_history": [],
                      "family_and_genetics_history": [],
                      "chief_complaint": ["Fatigue"]
                    }
                  }

                  Follow this structure for all provided inputs.
                  """
    )
