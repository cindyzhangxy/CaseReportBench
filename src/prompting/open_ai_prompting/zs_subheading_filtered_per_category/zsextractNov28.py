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
        prefix="""Return the result in JSON format with one key:
        is_IEM': A boolean value (`true` or `false`) indicating whether the condition is likely an IEM.
        Ensure:
        - The response is strictly in JSON format.
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

                """)


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
                  """)



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
                  """)

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
                  """)


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
                  """)


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
                  """)


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
                  """)


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
                  """)


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
                  """)


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
                  """)



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
                  """)



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
                  """)


# In[61]:


class History(Signature):
    """
    Extract detailed historical information from medical texts focusing on the patient's past medical and surgical history, history of present illness, social history, and family and genetics history, and chief complaint that brings patient to medical attention. This class is designed to capture comprehensive background data essential for diagnosis and treatment planning, explicitly excluding unrelated systemic conditions and treatment details.
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
                  """)
