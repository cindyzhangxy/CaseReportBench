#!/usr/bin/env python
# coding: utf-8

import dspy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import re
from dspy import (
    InputField,
    OutputField,
    Signature,
    ChainOfThought,
    Predict,
    Example,
    Assert, 
    Module
)
import ast
import json
import os
 

api_key =  os.getenv('OPENAI_API_KEY')

model="gpt-4o"
lm = dspy.LM(model=model, temperature=0.0,  max_tokens=2000, api_key=api_key)
dspy.configure(lm=lm)


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


class IEM(Signature):
    """
    Inborn errors of metabolism  are genetic disorders affecting the body's ability to metabolize nutrients, leading to various symptoms and lab abnormalities.
    They can be autosomal recessive, autosomal dominant, or X-linked. Inborn errors of metabolism involve failures in metabolic pathways for carbohydrates, fatty acids, and proteins.
    Although rare individually, Inborn errors of metabolism collectively occur in 1 in 2500 births and can present at any age. Determine if the input case text is likely an IEM based on the case report text. 
    Return output as a JSON format.
    """
    
    text = InputField(desc="Case report text")
    
    extract_info = OutputField(
        desc="Based on the provided text, determine if the medical condition described is an inborn errors of metabolism.",
        prefix="""Example output: {"Inborn errors of metabolism": 'Yes'}  
        Example Output 2: {"Inborn errors of metabolism": 'No'} 
        """
    )


class Pregnancy(Signature):
    """
    Extract detailed information from medical texts focusing on neonatal and maternal health outcomes and pregnancy complications. This class is designed to capture observations, health assessments, and conditions specific to pregnancy, explicitly excluding treatment details and any unrelated systemic conditions.
    """
    text = InputField(
        desc="Provide medical text that includes specific observations, health assessments, or conditions related to pregnancy, neonatal and maternal health."
    )

    extract_info = OutputField(
        desc="Organize pregnancy, neonatal, and maternal health information into a JSON-compliant dictionary.",
        prefix="""Based on the provided text, extract and return information directly relevant to pregnancy, neonatal, and maternal health in a structured dictionary format. Include the following top-level keys:
                  - 'neonatal_health': List conditions or observations directly related to the infant's health. Use an empty list [] if no pertinent information is available.
                  - 'maternal_health': List conditions or observations directly related to the mother's health. Use an empty list [] if no pertinent information is available.
                  - 'prengancy_test_imaging_exam': List all maternal and neonate lab tests, genetics tests, physical exam, and diangostic 
                  imaging and their respective positive and negative results. Use an empty list [] if no complications are found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the pregnancy, maternal or neonatal observations.
        
                  Example output for provided text containing relevant data:
                  {
                    "neonatal_health": ["Premature birth observed"],
                    "maternal_health": ["Gestational diabetes diagnosed"],
                    "prengancy_tests_image_exam": ["24 week ultrasound reveals normal fetal development"]
                  }
                  Example output if no relevant data is available:
                  {
                    "neonatal_health": [],
                    "maternal_health": [],
                    "pregnancy_tests_image_exam": []
                  }""")


class Vitals_Hema(Signature):
    """
    Extract vital signs and hematological observations from provided text, focusing solely on the measurement values and their units, and excluding any unrelated systemic conditions, treatments, or non-diagnostic content such as figures and tables.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc= "Organize vital sign and hematological information into a JSON-compliant dictionary, focusing on precise and clinically relevant data.", 
        prefix="""Based on the provided text, extract and return vital signs and hematological information in a structured dictionary format. Include the following top-level keys:
                  - 'temperature': List the temperature values combined with units if mentioned. Use an empty list [] if no temperature data is available.
                  - 'pulse': List pulse rate values combined with units if mentioned. Use an empty list [] if no pulse data is available.
                  - 'respiratory_rate': List respiratory rate values combined with units if mentioned. Use an empty list [] if no respiratory data is available.
                  - 'blood_pressure': List blood pressure readings, specifying both systolic and diastolic values combined with units if mentioned. Use an empty list [] if no blood pressure data is available.
                  - 'oxygen_saturation (SpO2)': List oxygen saturation values combined with units if mentioned. Use an empty list [] if no SpO2 data is available.
                  - 'hematological_conditions': List conditions or observations directly related to the the blood system, which includes components like red blood cells, white blood cells, platelets, blood vessels, bone marrow, lymph nodes, and the proteins involved in bleeding and clotting.
                  - 'hematology_tests_measurements': List all hematological related measurements such as hemoglobin, hematocrit, white blood cell count, platelet count, and any other relevant blood test results. Use an empty list [] if no hematology data is available.
                  Ensure all entries are verbatim quotations from or clear deductions from the text, without making assumptions. 
                  Focus on diagnostic clarity and ensure that the output is directly relevant to the reasons for the medical visit. 
                  Exclude reference to any figures or tables.  
                  Example output for provided text containing relevant data:
                  {
                    "temperature": ["37.5°C"],
                    "pulse": ["72 bpm"],
                    "respiratory Rate": ["16 breaths per minute"],
                    "blood_pressure": ["120/80 mm Hg"],
                    "oxygen_saturation (SpO2)": ["98%"],
                    "hematological_conditions": ["Diagnosed with anemia"]
                    "hematology_tests_measurements": ["Hemoglobin: 13.5 g/dL", 
                    "WBC count: 6,000 /µL",   "Platelet count: 250,000 /µL"]
                  }
                  Example output if no relevant data is available:
                  {
                    "temperature": [],
                    "pulse": [],
                    "respiratory_rate": [],
                    "blood_pressure": [],
                    "oxygen_saturation (SpO2)": [],
                    "hematological_conditions": []
                    "hematology_tests_measurements": []
                  }""")



class Immune(Signature):
    """
    Extract immunological, serological, oncological observations, and related physical exam or imaging findings from provided text, focusing on conditions, cell counts, biomarkers, and excluding any unrelated systemic conditions, treatments, or non-diagnostic content such as figures and tables.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc="Organize immunological, oncological information, and related exam or imaging findings into a JSON-compliant dictionary, focusing on precise and clinically relevant data.",
        prefix="""Based on the provided text, extract and return immunological, oncological information, and related physical exam or imaging findings in a structured dictionary format. Include the following top-level keys:
                  - 'immunity_conditions': List the current cofirmed immunological conditions.
                  - 'immune_cell_counts': Detailed counts of immune cells like lymphocytes, monocytes, etc.
                  - 'immunophenotyping': Results from immunophenotyping tests detailing the types and statuses of immune cells.
                  - 'autoimmune_markers': List markers indicative of autoimmune diseases.
                  - 'inflammatory_markers': Include levels of inflammatory markers such as CRP, ESR, or procalcitonin.
                  - 'serology': Results from serological tests indicating the presence of specific antibodies or antigens.
                  - 'infectious_disease_status': Document current infectious diseases confirmed through clinical evaluation or laboratory testing.
                  - 'malignancy_biomarkers': List specific biomarkers that are used to diagnose or monitor malignancies, such as tumor markers or genetic mutations.
                  - 'malignancy_condition': List confirmed current diagnosis and conditions related to malignancies, detailing the type of cancer and any relevant clinical observations.
                  - 'immunology_exam_image': Include any positive or negative findings if available from physical exams or imaging studies that are pertinent to the immunological, infectious disease, or oncological conditions.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of immunological or infectious diseases or oncological observations.
                  
                  Example output for provided text containing relevant data:
                  {
                    "immunity_conditions": ["Chronic lymphocytic leukemia diagnosed"],
                    "immune_cell_counts": ["Lymphocytes: 5000 cells/µL"],
                    "immunophenotyping": ["CD4 cell count: 350 cells/µL"],
                    "autoimmune_markers": ["ANA positive"],
                    "inflammatory_markers": ["CRP: 10 mg/L"],
                    "serology": ["HIV antibodies detected"],
                    "infectious_disease_status": ["Confirmed COVID-19 infection"],
                    "malignancy_biomarkers": ["CA-125 elevated indicating possible ovarian cancer"],
                    "malignancy_condition": ["Diagnosed with Stage II breast cancer"],
                    "immunology_exam_image": ["Ultrasound shows enlarged lymph nodes", "MRI scan detects multiple myeloma lesions"]
                  }
                  Example output if no relevant data is available:
                  {
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
                  }""")

class Neuro(Signature):
    """
    Extract detailed information from medical texts focusing on assessments related to neurological and cognitive functions, as well as conditions specifically affecting the head. This class is designed to capture observations, diagnostic results, and imaging findings specific to these areas, explicitly excluding treatment details and any unrelated systemic conditions.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc="Organize neurological, cognitive, and head information into a JSON-compliant dictionary, focusing exclusively on relevant diagnostic data. ",
        prefix="""Based on the provided text, extract and return information directly relevant to neurological and cognitive functions as well as head-related conditions in a structured dictionary format. Include the following top-level keys:
                  - 'neurological': List observations and conditions directly related to the neurological system. Use an empty list [] if no pertinent information is available.
                  - 'cognitive': List observations and conditions directly related to cognitive functions. Use an empty list [] if no pertinent information is available.
                  - 'Neuro_Tests_Imaging_Exam': List descriptions of tests, measurements, physical exam, and diagnostic imaging specific to the neurological and cognitive areas, detailing both positive and negative findings if available. Use an empty list [] if no applicable data is found.
                  Ensure all entries are verbatim quotes from or clear deductions based from the provided text, without making assumptions or inference.  
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures and tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the neurological observations.
                  
                  Example output for provided text containing relevant data:
                  {
                    "neurological": ["Increased intracranial pressure observed"],
                    "cognitive": ["Impaired short-term memory noted"],
                    "Neuro_Tests_Imaging": ["MRI Brain: Evidence of cerebral atrophy"]
                  }
                  Example output if no relevant data is available:
                  {
                    "neurological": [],
                    "cognitive": [],
                    "Neuro_tests_image_exam": []
                  }""")


class EENT(Signature):
    """
    Extract specific EENT (Eyes, Ears, Nose, and Throat) information from medical text, focusing on conditions and symptoms directly affecting these areas without making inferences about related systemic conditions.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc="Extract and organize EENT information into a JSON-compliant dictionary, focusing on observations and symptoms directly relevant to EENT structures. Exclude all treatment details and conditions not directly involving the EENT structures.",
        prefix="""Based on the provided text, extract and return information directly relevant to EENT conditions in a structured dictionary format. Include observations or symptoms for 'eyes', 'ears', 'nose', and 'throat', using the following top-level keys:
                  - 'eyes': Include only direct observations or symptoms related to eye conditions. Use an empty list [] if no information is available.
                  - 'ears': Include only direct observations or symptoms related to ear conditions. Use an empty list [] if no information is available.
                  - 'nose': Include only direct observations or symptoms related to nose conditions. Use an empty list [] if no information is available.
                  - 'throat': Include only direct observations or symptoms related specifically to throat conditions. Use an empty list [] if no information is available.
                  - 'EENT_tests_image_exam': Include a sub-dictionary for EENT-specific tests, measurements, physical exam, and diagnostic imaging, detailing their respective positive and negative findings if available. Use an empty list [] if no relevant tests or imaging results are mentioned.
                  Ensure all entries are verbatim or clear deductions from the provided text without making assumptions. 
                  all entries should focus on diagnostic clarity and exclude any treatment details, references to figures or tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the ears, eyes, nose and throat observations.
                  
                  Here is an example of the output format:
{
  "eyes": ["Detailed description of eye symptoms from the text"],
  "ears": ["Detailed description of ear symptoms from the text"],
  "nose": ["Detailed description of nasal symptoms from the text"],
  "throat": ["Detailed description of throat symptoms from the text"],
  "EENT_tests_image_exam": {"Test Name": "Specific findings from the test relevant to EENT"}
}

 Example output if no relevant data is available:
                  {
                    "eyes": [],
                    "ears": [],
                    "nose": [],
                    "throat": [],
                    "vascular": [],
                    "EENT_tests_image_exam": []
""")


class CVS(Signature):
    """
    Extract specific information about the cardiovascular and vascular systems from medical texts, focusing on observable or reported data related to heart and blood vessel functions without including treatment details or unrelated systemic conditions.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc="Extract and organize cardiovascular and vascular system information into a JSON-compliant dictionary, focusing solely on relevant details. Exclude any treatment data or unrelated systemic condition references.",
        prefix="""Based on the provided text, extract and return relevant information in a structured dictionary format. Include the following top-level keys:
                  - 'cardiac': List observations, signs, symptoms, or conditions directly related to the heart and its immediate functions also include cardiocerebral conditions such as stroke. Use an empty list [] if no pertinent information is available.
                  - 'vascular': List observations, signs, symptoms, or conditions directly related to the blood vessels and circulatory system. Use an empty list [] if no applicable data is found.
                  - 'CVS_tests_image_exam': List all cardiovascular lab tests, genetics tests, physical exam, and diangostic imaging and their respective positive and negative results if available. Use an empty list [] if no complications are found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the cardiovascular observations.
        
                  {
                    "cardiac": ["Elevated heart rate observed during stress test"],
                    "vascular": ["Visible varicose veins", "Signs of peripheral arterial disease"],
                    "CVS_tests_image_exam": ["Echocardiogram shows mild hypertrophy", "Carotid ultrasound revealed Plaque buildup noted", "Endocardiogram was performed"]
                  }
                  Example output if no relevant data is available:
                  {
                    "cardiac": [],
                    "vascular": [],
                    "CVS_tests_image_exam": []
                  }""")


class RESP(Signature):
    """
    Extract specific information about the respiratory system from medical texts, focusing on observable or reported data related to respiratory functions without including treatment details or unrelated systemic conditions.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc="Extract and organize respiratory system information into a JSON-compliant dictionary, focusing solely on relevant details. Exclude any treatment data or unrelated systemic condition references.",
        prefix="""Based on the provided text, extract and return relevant respiratory system information in a structured dictionary format. Include the following top-level keys:
                  - 'respiratory': List observations, signs, symptoms, or conditions directly related to the respiratory system. Use an empty list [] if no pertinent information is available.
                  - 'RESP_tests_image_exam': List descriptions of tests, measurements, physical exams, and diagnostic imaging specific to the respiratory system, including both positive and negative findings. Use an empty list [] if no applicable data is found. 
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the respiratory observations.
        
                  Example output for provided text containing relevant data:
                  {
                    "respiratory": ["Increased respiratory rate observed during examination"],
                    "RESP_tests_image_exam": ["Spirometry: Reduced lung capacity", "Chest X-ray: No visible abnormalities"]
                  }
                  Example output if no relevant data is available:
                  {
                    "respiratory": [],
                    "RESP_tests_image_exam": []
                  }""")
    

class GI(Signature):
    """
    Extract specific information about the gastrointestinal system from medical texts, focusing on observable or reported data related to gastrointestinal functions without including treatment details or unrelated systemic conditions.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc="Extract and organize gastrointestinal system information into a JSON-compliant dictionary, focusing solely on relevant details. Exclude any treatment data or unrelated systemic condition references.",
        prefix="""Based on the provided text, extract and return relevant gastrointestinal system information in a structured dictionary format. Include the following top-level keys:
                  - 'gastrointestinal': List observations, signs, symptoms, or conditions directly related to the gastrointestinal system. Use an empty list [] if no pertinent information is available.
                  - 'GI_tests_image_exam': List descriptions of tests, measurements, physical exams and diagnostic imaging specific to the gastrointestinal system, including both positive and negative findings if available. Use an empty list [] if no applicable data is found.
                  Ensure all entries are verbatim or clear deductions from the provided text without making assumptions. 
                  All etnries should focus on diagnostic clarity. Exclude any treatment details and exclude references to figures or tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the gastrointestinal observations.
               
                  Example output for provided text containing relevant data:
                  {
                    "gastrointestinal": ["Increased abdominal discomfort noted during examination"],
                    "GI_tests_image_exam": ["Colonoscopy: Evidence of polyps", "Abdominal Ultrasound: Normal liver and gallbladder morphology"]
                  }
                  Example output if no relevant data is available:
                  {
                    "gastrointestinal": [],
                    "GI_tests_image_exam": []
                  }""")


class GU(Signature):
    """
    Extract detailed information from medical texts focusing on assessments related to the genital and urinary systems. This class is designed to capture observations, diagnostic results, and imaging findings specific to these systems, explicitly excluding treatment details and any unrelated systemic conditions.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc="Organize genitourinary system information into a JSON-compliant dictionary, focusing solely on relevant diagnostic data. Exclude any treatment details and non-diagnostic content such as figures or unrelated systemic conditions.",
        prefix="""Based on the provided text, extract and return the relevant genitourinary system information in a structured dictionary format. Separate the information into the following top-level keys:
                  - 'urinary': List observations and conditions directly related to the urinary tract. Use an empty list [] if no pertinent information is available.
                  - 'genital': List observations and conditions directly related to the genital organs. Use an empty list [] if no pertinent information is available.
                  - 'GU_tests_image_exam':  List descriptions of tests, measurements, physical exams and diagnostic imaging specific to the genital and urinary systems, detailing both positive and negative findings if available. 
                  Use an empty list [] if no applicable data is found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the genitourinary observations.          
                  Example output for provided text containing relevant data:
                  {
                    "urinary": ["Bladder was full"],
                    "genital": ["Prostate enlargement noted"],
                    "GU_tests_image_exam": ["Ultrasound Kidney: No stones detected", "Bladder Ultrasound: Normal bladder wall thickness"]
                  }
                  Example output if no relevant data is available:
                  {
                    "urinary": [],
                    "genital": [],
                    "GU_tests_image_exam": []
                  }""")


class DERM(Signature):
    """
    Extract detailed information from medical texts focusing on assessments related to the skin, facial features, and breast conditions. This class is designed to capture observations, diagnostic results, and imaging findings specific to dermatological assessments, explicitly excluding treatment details and any unrelated systemic conditions.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc="Organize dermatological information into a JSON-compliant dictionary, focusing solely on relevant diagnostic data. Exclude any treatment details and non-diagnostic content such as figures or unrelated systemic conditions.",
        prefix="""Based on the provided text, extract and return the relevant dermatological information in a structured dictionary format. Separate the information into the following top-level keys:
                  - 'skin_conditions': List observations and conditions directly related to the skin. Use an empty list [] if no pertinent information is available.
                  - 'facial_features': List observations and conditions directly related to the facial features. Use an empty list [] if no pertinent information is available.
                  - 'breast_conditions': List observations and conditions directly related to the breasts. Use an empty list [] if no pertinent information is available.
                  - 'derm_breasts_facial_tests_image_exam': List descriptions of tests, measurements, physical exams and diagnostic imaging specific to dermatological, breasts and facial feature assessments, detailing both positive and negative findings if available. Use an empty list [] if no applicable data is found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the dermatology or facial or breasts observations.
        
                  
                  Example output for provided text containing relevant data:
                  {
                    "skin_conditions": ["Psoriasis noted", "Severe acne observed"],
                    "facial_features": ["Rosacea on cheeks"],
                    "breast_conditions": ["breast looks normal"],
                    "derm_breasts_facial_tests_image_exam": ["Dermatoscopy: Melanocytic nevus identified", "Skin biopsy: Basal cell carcinoma confirmed", "mammography": "unremarkable fidings"]
                  }
                  Example output if no relevant data is available:
                  {
                    "skin_conditions": [],
                    "facial_features": [],
                    "breast_conditions": [],
                    "derm_breasts_facial_tests_image_exam": []
                  }""")


class MSK(Signature):
    """
    Extract detailed information from medical texts focusing on assessments related to the musculoskeletal system, specifically targeting muscle and skeletal structures separately. This class is designed to capture observations, diagnostic results, and imaging findings specific to muscles and bones, explicitly excluding treatment details and any unrelated systemic conditions.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc="Organize musculoskeletal system information into a JSON-compliant dictionary, focusing solely on relevant diagnostic data for muscles and skeletal structures. Exclude any treatment details and non-diagnostic content such as figures or unrelated systemic conditions.",
        prefix="""Based on the provided text, extract and return the relevant musculoskeletal system information in a structured dictionary format. Organize the information into the following top-level keys:
                  - 'muscle': List observations and conditions directly related to muscle health and function. Use an empty list [] if no pertinent muscle information is available.
                  - 'skeletal': List observations and conditions directly related to the skeletal system, including bones and joints. Use an empty list [] if no pertinent skeletal information is available.
                  - 'MSK_tests_image_exam': List descriptions of tests, measurements, and diagnostic imaging that are specific to the musculoskeletal system, covering both muscles and bones, detailing both positive and negative findings if available. Use an empty list [] if no applicable data is found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the musculoskeletal observations.
        
                  Example output for provided text containing relevant data:
                  {
                    "muscle": ["Muscle stiffness and pain reported"],
                    "skeletal": ["Bone density reduction noted", "Joint swelling observed"],
                    "MSK_tests_image_exam": ["MRI: Ligament tear detected", "Bone scan: Signs of osteoporosis"]
                  }
                  Example output if no relevant data is available:
                  {
                    "muscle": [],
                    "skeletal": [],
                    "MSK_tests_image_exam": []
                  }""")



class LYMPH(Signature):
    """
    Extract detailed information from medical texts focusing on assessments related to the lymphatic system. This class is designed to capture observations, diagnostic results, and imaging findings specific to components of the lymphatic system, explicitly excluding treatment details and any unrelated systemic conditions.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc="Organize lymphatic system information into a JSON-compliant dictionary, focusing solely on relevant diagnostic data. Exclude any treatment details and non-diagnostic content such as figures or unrelated systemic conditions.",
        prefix="""Based on the provided text, extract and return the relevant lymphatic system information in a structured dictionary format. 
        Exclude reference to figures, tables or citations. Separate the information into the following top-level keys:
                  - 'adenoid': List observations and conditions directly related to adenoids. Use an empty list [] if no pertinent information is available.
                  - 'tonsils': List observations and conditions directly related to the tonsils. Use an empty list [] if no pertinent information is available.
                  - 'lymphatic_tissues': List observations and conditions directly related to general lymphatic tissues. Use an empty list [] if no pertinent information is available.
                  - 'lymph_nodes': List observations and conditions directly related to lymph nodes. Use an empty list [] if no pertinent information is available.
                  - 'thymus': List observations and conditions directly related to the thymus. Use an empty list [] if no pertinent information is available.
                  - 'bone_marrow': List observations and conditions directly related to bone marrow. Use an empty list [] if no pertinent information is available.
                  - 'spleen': List observations and conditions directly related to the spleen. Use an empty list [] if no pertinent information is available.
                  - 'immune_cells': List observations and conditions directly related to immune cells. Use an empty list [] if no pertinent information is available.
                  - 'Lymphatic_tests_image_exam': List descriptions of tests, measurements, physical exams and diagnostic imaging for any part of the lymphatic system, detailing both positive and negative findings if available. Use an empty list [] if no applicable data is found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the lymphatic system observations.
        
                  Example output for provided text containing relevant data:
                  {
                    "adenoid": ["Enlarged adenoids observed"],
                    "tonsils": ["Tonsillitis diagnosed"],
                    "lymphatic_tissues": ["Signs of lymphedema noted"],
                    "lymph_nodes": ["Lymphadenopathy in cervical nodes"],
                    "thymus": ["Thymus hyperplasia found"],
                    "bone_marrow": ["Bone marrow biopsy shows increased cellularity"],
                    "spleen": ["Splenomegaly detected"],
                    "immune_cells": ["Increased leukocytes in blood test"],
                    "Lymphatic_tests_image_exam": ["PET scan: Abnormal lymph node activity"]
                  }
                  Example output if no relevant data is available:
                  {
                    "adenoid": [],
                    "tonsils": [],
                    "lymphatic_tissues": [],
                    "lymph_nodes": [],
                    "thymus": [],
                    "bone_marrow": [],
                    "spleen": [],
                    "immune_cells": [],
                    "Lymphatic_tests_image_exam": []
                  }""")



class ENDO(Signature):
    """
    Extract detailed information from medical texts focusing on assessments related to the endocrine system. This class is designed to capture observations, diagnostic results, and imaging findings specific to endocrine glands, explicitly excluding treatment details and any unrelated systemic conditions.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc="Organize endocrine system information into a JSON-compliant dictionary, focusing solely on relevant diagnostic data. Exclude any treatment details and non-diagnostic content such as figures or unrelated systemic conditions.",
        prefix="""Based on the provided text, extract and return the relevant endocrine system information in a structured dictionary format. Separate the information into the following top-level keys:
                  - 'endocrine_glands': List observations and conditions directly related to specific endocrine glands such as the thyroid, pancreas, adrenal glands, pituitary gland, and others. Use an empty list [] if no pertinent information is available.
                  - 'Endocrine_tests_image_exam': List descriptions of tests, measurements, physical exams, and diagnostic imaging specific to the endocrine system, including both positive and negative findings if available. Use an empty list [] if no applicable data is found.
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the endocrine system observations.
        
                  Example output for provided text containing relevant data:
                  {
                    "endocrine_glands": ["Thyroid enlargement noted", "Adrenal insufficiency observed"],
                    "Endocrine_tests_image_exam": ["Thyroid function test results: Elevated TSH", "CT scan: Adrenal mass detected"]
                  }
                  Example output if no relevant data is available:
                  {
                    "endocrine_glands": [],
                    "Endocrine_tests_image_exam": []
                  }""")


# In[61]:


class History(Signature):
    """
    Extract detailed historical information and cheif complaints from medical texts focusing on the patient's past medical and surgical history, history of present illness, social history, and family and genetics history, and chief complaint that brings patient to medical attention. This class is designed to capture comprehensive background data essential for diagnosis and treatment planning, explicitly excluding unrelated systemic conditions and treatment details.
    """
    text = InputField(
        desc="Medical text."
    )

    extract_info = OutputField(
        desc="Organize historical information into a JSON-compliant dictionary, focusing solely on relevant historical data. Exclude any treatment details and non-diagnostic content such as figures or unrelated systemic conditions.",
        prefix="""Based on the provided text, extract and return the relevant historical information in a structured dictionary format. Separate the information into the following top-level keys:
                  - 'past_medical_history': List conditions and previous diagnoses related to the patient's past medical history. Use an empty list [] if no pertinent information is available.
                  - 'past_surgical_history': List past surgical interventions and outcomes. Use an empty list [] if no relevant surgical history is available.
                  - 'history_of_present_illness': List the chief complaint and detail the chronological development of the patient's current complaints and symptoms. Use an empty list [] if no detailed current illness history is provided.
                  - 'social_history': Include relevant lifestyle factors such as smoking, alcohol use, occupation, and living conditions. Use an empty list [] if no social history is available.
                  - 'family_and_genetics_history': List any known genetic conditions or diseases prevalent in the patient's family that might affect the patient's health. Use an empty list [] if no family or genetic history is available.
                   - 'chief complaint': List any known chief complaint that brought patient to seek medical attention. Use an empty list [] if no information. 
                  All entries are verbatim quotes from or clear deductions based on the provided text, without making assumptions.
                  All entries should focus on diagnostic clarity and exclude any treatment details, references to figures, tables.
                  For all entries, do not infer or suggest possible underlying or associated conditions that are not explicitly part of the historical information provided.
        
                  Example output for provided text containing relevant data:
                  {
                    "past_medical_history": ["Diagnosed with hypertension", "Previous myocardial infarction"],
                    "past_surgical_history": ["Appendectomy in 2010", "Knee replacement in 2018"],
                    "history_of_present_illness": ["Gradual onset of chest pain over the past two months"],
                    "social_history": ["Smoker for 20 years, 10 cigarettes a day", "Works in construction"],
                    "family_and_genetics_history": ["Father had colon cancer", "Sister diagnosed with breast cancer at age 50"],
                    "chief_complaint":["Patient was brought to ER after the first episode of acute chest pain and hemoptysis"]
                  }
                  Example output if no relevant data is available:
                  {
                    "past_medical_history": [],
                    "past_surgical_history": [],
                    "history_of_present_illness": [],
                    "social_history": [],
                    "family_and_genetics_history": [],
                    "chief_complaint": [] 
                  }""")
