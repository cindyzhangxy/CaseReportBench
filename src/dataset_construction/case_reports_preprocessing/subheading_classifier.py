import pandas as pd
import json
from glob import glob
import re
import altair as alt 


list_of_dict = []
for file in glob("../../../NLP_Enhanced_IEM/data/processed/case_report_text/PMC*.json"):
    with open(file, 'r', encoding="utf-8") as f:
        list_of_dict.extend(json.load(f)) 

list_of_case = []
for case in list_of_dict:
    if case: 
        if case["case"]:  
            list_of_case.append(case)

def remove_fig_digit(text):
     # Pattern to cover variations of figure references and standalone digits
    pattern = r'\(\s*[Ff][Ii][Gg]\.?\s*\d+\w?\s*\)|\(\s*\d+\s*(,\s*\d+\s*)*\)'
    # Remove patterns found
    new_text = re.sub(pattern, '', text)
    # Normalize different types of spaces and ensure regular spacing
    new_text = new_text.replace('\u2009', ' ').replace('\xa0', ' ')
    # Normalize space by replacing multiple spaces with a single space
    new_text = re.sub(r'\s+', ' ', new_text)
    return new_text.strip()

for item in list_of_case:
    for k, v in item["case"].items():
        if k and v: 
        # Corrected variable name in list comprehension
            item[k] = [remove_fig_digit(s) for s in v]

list_of_case = [case for case in list_of_case if case["title"]!='']

selected_key_general = [
    "patient", "case", "presentation", "complaint", "report" 
]
selected_keys_vitals = ["vital"]

selected_keys_NEURO =  ["brain", "cereb", "neuro", "nerv", "sensory", "memory", "reflex", "behavior", 
    "cogni", "psych", "skull", "encepha", "mening", "spine"
] 

selected_keys_EENT = [
    "oral", "mouth", "tongue", "eye", "ear", "nose", "pharynx", "larynx", "throat", 
    "neck", "eyebrow", "ocular", "rhin", "opt", "ophthalm", "vision", "sensory", 
    "audi", "visual"
]

selected_keys_respiratory = [
    "respiratory", "chest", "bronch", "lung", "alveol", "pleur", "trache", 
    "pneum", "pulmo", "thorac"
]

selected_keys_cardiovascular = [
    "cardi", "heart", "CVS", "vascul", "thrombo", "athero"
]

selected_keys_gastrointestinal = [
     "gastr", "GI", "liver", "gallbladder", "omentum", "peritoneum", "hepat", "abdo", 
    "esoph", "stomach", "intestin", "colo", "rect", "anal", "duoden", "jejunum", "ileum", 
    "pancrea", "sigmoid", "appendix", "caecum", "biliary"
]

selected_keys_genitourinary = [
    "GU", "genitourinary", "prosta", "ureth", "bladder", "renal", "kidney", "cyst", "nephro", "groin",
    "genital", "ovar", "testi", "vagin", "uter", "cervix", "scrot"
]

selected_keys_dermatological = [
    "skin", "derm", "breast", "cutane", "melan"
]

selected_keys_endocrine = [
    "endocrin", "pituit", "adrenal", "gland", "pancrea", "thyroid"
]

selected_keys_musculoskeletal = [
    "ortho", "skeletal", "musc", "arm", "leg", "hand", "feet", "joint"
]

selected_keys_lymph = [
    "immun", "adenoid", "tonsil", "lymph", "thymus", "appendix", "marrow", "spleen"
]

selected_keys_imaging = [
    "radi", "X-Ray", 
    "sonograph", "echo", "ultrasound",
    "fluoro",
    "densitometry", "DEXA", "DXA",
    "mammo",
    "tomography",
    "magnet",
    "nucle",
    "angio"
]

selected_keys_test = [
    "hemo", "hema", "chemi", "A1C", "albumin", "creatinine", "lipid", "glucose",
     "thyroid", "diabetes", "apolipoprotein", "microbi", "myco", "dermatophytes", "NAAT", 
     "vaginitis", "serology",  "electrocardio", "pressure", "holter", "FIT", "Lactose", "Semen", 
    "fecal",  "Creatine", "fluid","electrolyte"
    
]

selected_keys_general_info = [
    "patient", "case", "presentation", "complaint", "report", "observation", "exam", "pheno", "measure", "analysis"
]

selected_keys_PE = [
    "physical", "inspect", "palpat", "percuss", "ascultat", 
]

selected_keys_maternal = [
    "Neonate", "partum", "pregnan", "natal"
]

selected_keys_history = [
    "history"
]


def find_matching_keys(report, selected_keys):
    """This function finds all keys in report["case"] that contain any of the selected_keys as substrings."""
    matching_keys = []
    for key in report["case"]:   
        for selected_key in selected_keys:
            if selected_key.lower() in key.lower():
                matching_keys.append(key)   
                break  # Once a match for this key is found with one of the roots, stop checking others to avoid duplication
    return matching_keys


keys_vitals = []
keys_NEURO = []
keys_EENT = []
keys_respiratory = []
keys_cardiovascular = []
keys_gastrointestinal = []
keys_genitourinary = []
keys_dermatological = []
keys_endocrine = []
keys_musculoskeletal = []
keys_lymph = []
keys_imaging = []
keys_test = []
keys_maternal = []
keys_general_info = []
keys_PE = []
keys_history = []

for report in list_of_case:
    keys_NEURO.extend(find_matching_keys(report, selected_keys_vitals))
    keys_NEURO.extend(find_matching_keys(report, selected_keys_NEURO))
    keys_EENT.extend(find_matching_keys(report, selected_keys_EENT))
    keys_respiratory.extend(find_matching_keys(report, selected_keys_respiratory))
    keys_cardiovascular.extend(find_matching_keys(report, selected_keys_cardiovascular))
    keys_gastrointestinal.extend(find_matching_keys(report, selected_keys_gastrointestinal))
    keys_genitourinary.extend(find_matching_keys(report, selected_keys_genitourinary))
    keys_dermatological.extend(find_matching_keys(report, selected_keys_dermatological))
    keys_endocrine.extend(find_matching_keys(report, selected_keys_endocrine))
    keys_musculoskeletal.extend(find_matching_keys(report, selected_keys_musculoskeletal))
    keys_lymph.extend(find_matching_keys(report, selected_keys_lymph))
    keys_imaging.extend(find_matching_keys(report, selected_keys_imaging))
    keys_test.extend(find_matching_keys(report, selected_keys_test))
    keys_general_info.extend(find_matching_keys(report, selected_keys_general_info))
    keys_PE.extend(find_matching_keys(report, selected_keys_PE))
    keys_maternal.extend(find_matching_keys(report, selected_keys_maternal))
    keys_history.extend(find_matching_keys(report, selected_keys_history))
    
keys_NEURO = list(set(keys_NEURO))
keys_EENT = list(set(keys_EENT))
keys_respiratory = list(set(keys_respiratory))
keys_cardiovascular = list(set(keys_cardiovascular))
keys_gastrointestinal = list(set(keys_gastrointestinal))
keys_genitourinary = list(set(keys_genitourinary))
keys_dermatological = list(set(keys_dermatological))
keys_endocrine = list(set(keys_endocrine))
keys_musculoskeletal = list(set(keys_musculoskeletal))
keys_lymph = list(set(keys_lymph))
keys_imaging = list(set(keys_imaging))
keys_test = list(set(keys_test))
keys_maternal = list(set(keys_maternal))
keys_general_info = list(set(keys_general_info))
keys_PE = list(set(keys_PE))
keys_history = list(set(keys_history))

subheading_dict = {
    "Vitals": keys_vitals,
    "NEURO": keys_NEURO,
    "EENT": keys_EENT,
    "RESP": keys_respiratory,
    "CVS": keys_cardiovascular,
    "GI": keys_gastrointestinal,
    "GU": keys_genitourinary,
    "Derm": keys_dermatological,
    "Endocrine": keys_endocrine,
    "MSK": keys_musculoskeletal,
    "Lymphatic": keys_lymph,
    "Imaging": keys_imaging,
    "Lab": keys_test,
    "General": keys_general_info,
    "Physical Exam": keys_PE,
    "Maternal": keys_maternal,
    "History": keys_history
}

master_key_set = set(keys_NEURO) | set(keys_EENT) | set(keys_respiratory) | \
                 set(keys_cardiovascular) | set(keys_gastrointestinal) | set(keys_genitourinary) | \
                 set(keys_dermatological) | set(keys_endocrine) | set(keys_musculoskeletal) | \
                 set(keys_lymph) | set(keys_imaging) | set(keys_test) | \
                 set(keys_maternal) | set(keys_general_info) | set(keys_PE) | set(keys_history)
master_key_list = list(master_key_set)

with open("./output/subheading.json", "w", encoding="utf-8") as f:
    json.dump(subheading_dict, f)

new_case_list = []

for report in list_of_case:
    new_report = {"title": report["title"]}  # Initialize new_report with the title
    matched_keys = find_matching_keys(report, master_key_list) 
    
    
    if matched_keys:     
        new_report['case'] = {key: report['case'][key] for key in set(matched_keys) if key in report['case'] and report['case'][key]}
        
        # Add the new report to the list if there is at least one valid case entry
        if new_report['case']:
            new_case_list.append(new_report)

for report in new_case_list:
    # for each report, transform  the 'case' dictionary
    report["case"] = {k: " ".join(v) for report in new_case_list for k, v in report["case"].items()}

with open("./output/filtered_case_report_list.json", "w", encoding="utf-8") as f:
    json.dump(new_case_list, f)

key = {k: set(v) for k, v in subheading.items()}

def combine_sets(set_names):
    """Combine multiple sets into one by union operation."""
    combined_set = set()
    for name in set_names:
        if name in key:
            combined_set.update(key[name])
        else:
            print(f"Warning: Set {name} not found")
    return combined_set

def get_excluded_sets(excluded_names):
    """Retrieve all elements from excluded categories."""
    excluded_set = set()
    for name in excluded_names:
        if name in key:
            excluded_set.update(key[name])
        else:
            print(f"Warning: Set {name} not found")
    return excluded_set

def combine_and_filter(keys, exclude=None):
    """Combine sets based on keys and optionally remove excluded items."""
    result = set().union(*(key[k] for k in keys))
    if exclude:
        result -= exclude
    return result

def calculate_exclusive(category_name, all_names, total_set):
    """Calculate exclusive elements of a category within the total set."""
    other_categories = [name for name in all_names if name != category_name]
    combined_others = combine_and_filter(other_categories)
    category_set = key.get(category_name, set())

    return (category_set - combined_others).intersection(total_set)

combined_names = ["General", "Physical Exam", "Lab", "Imaging"]
all_names = ['Vitals', 'NEURO', 'EENT', 'RESP', 'CVS', 'GI', 'GU', 'Derm', 'Endocrine', 'MSK', 'Lymphatic', 'Imaging', 'Lab', 'General', 'Physical Exam', 'Maternal', 'History']

all = combine_sets(all_names)

general_pathopyisology = set([heading for heading in all 
                 if ("pathophy" in heading.lower() or "mechanism" in heading.lower() or "guideline" in heading.lower())
                 and "presentation" not in heading.lower()
                 and "patient" not in heading.lower()
                 and "case" not in heading.lower()
                 and "diagnos" not in heading.lower()
                 and "exam" not in heading.lower()
                 and "imag" not in heading.lower()
                ])

epidemiology = set([heading for heading in all
                 if ("epidemiolog" in heading.lower() or "incidence" in heading.lower() or "prevalence" in heading.lower())
                 and "presentation" not in heading.lower()
                 and "patient" not in heading.lower()
                 and "case" not in heading.lower()
                 and "diagnos" not in heading.lower()
                 and "exam" not in heading.lower()
                 and "imag" not in heading.lower()
                ])

questions = set([heading for heading in all 
                 if ("question" in heading.lower() or "?" in heading.lower())
                 and "presentation" not in heading.lower()
                 and "patient" not in heading.lower()
                 and "case" not in heading.lower()
                 and "diagnos" not in heading.lower()
                 and "exam" not in heading.lower() 
                 and "imag" not in heading.lower()
                ])

contribution = set([heading for heading in all
                 if ("contribu" in heading.lower() or "department" in heading.lower() 
                     or "professor" in heading.lower() or "disclo" in heading.lower() 
                     or "fund" in heading.lower() or "acknowledge" in heading.lower()
                     or "md" in heading.lower() or "ethic" in heading.lower() or "editor" in heading.lower()
                     or "guarantor")
                 and "presentation" not in heading.lower()
                 and "patient" not in heading.lower()
                 and "case" not in heading.lower()
                 and "diagnos" not in heading.lower()
                 and "exam" not in heading.lower()
                 and "imag" not in heading.lower()
                ])

statistics = set([heading for heading in all 
                 if ("statist" in heading.lower() or "data" in heading.lower() or "bioinformatics" in heading.lower())
                 and "presentation" not in heading.lower()
                 and "patient" not in heading.lower()
                 and "case" not in heading.lower()
                 and "diagnos" not in heading.lower()
                 and "exam" not in heading.lower()   
                 and "imag" not in heading.lower()
                ])

supplementary = set([heading for heading in all
                 if ("supplem" in heading.lower() or "figure" in heading.lower() or "table" in heading.lower() or "appendix" in heading.lower()
                    or "video" in heading.lower() or "online" in heading.lower())
                 and "presentation" not in heading.lower()
                 and "patient" not in heading.lower()
                 and "case" not in heading.lower()
                 and "diagnos" not in heading.lower()
                 and "exam" not in heading.lower()   
                 and "imag" not in heading.lower()
                ])

miscellaneous = set([heading for heading in all
                 if ("approval" in heading.lower() or "ethic" in heading.lower() or "publication" in heading.lower() 
                    or "learn" in heading.lower() or "lesson" in heading.lower() or "search" in heading.lower()
                    or "literature" in heading.lower() or "pubmed" in heading.lower() or "years" in heading.lower())
                 and "presentation" not in heading.lower()
                 and "patient" not in heading.lower()
                 and "case" not in heading.lower()
                 and "diagnos" not in heading.lower()
                 and "exam" not in heading.lower()   
                 and "imag" not in heading.lower()
                ])

treatment = set([heading for heading in all 
                 if ("interve" in heading.lower() or "treat" in heading.lower() or "therap" in heading.lower() 
                     or "medic" in heading.lower() or "restore" in heading.lower() or "follow-up" in heading.lower())
                 and "presentation" not in heading.lower()
                 and "patient" not in heading.lower()
                 and "case" not in heading.lower()
                 and "diagnos" not in heading.lower()
                 and "exam" not in heading.lower()
                 and "imag" not in heading.lower()
                 and "find" not in heading.lower()
                 and "history" not in heading.lower()
                 and "test" not in heading.lower()
                 and "fuction" not in heading.lower()
                 
                ])
adverse_effect = set([heading for heading in all 
                 if ("adverse" in heading.lower() or "complication" in heading.lower())
                 and "presentation" not in heading.lower()
                 and "patient" not in heading.lower()
                 and "case" not in heading.lower()
                 and "diagnos" not in heading.lower()
                 and "exam" not in heading.lower()
                 and "imag" not in heading.lower()     
                ])

surgical = set([heading for heading in all
                 if ("surgical" in heading.lower() or "surgery" in heading.lower())
                 and "presentation" not in heading.lower()
                 and "patient" not in heading.lower()
                 and "case" not in heading.lower()
                 and "diagnos" not in heading.lower()
                 and "exam" not in heading.lower()
                 and "imag" not in heading.lower()
                ])

excluded_categories = {
    "questions": list(questions),
    "contribution": list(contribution),
    "statistics": list(statistics),
    "supplementary": list(supplementary),
    "general_pathophysiology": list(general_pathopyisology),
    "epidemiology": list(epidemiology),
    "treatment": list(treatment),
    "adverse_effect": list(adverse_effect),
    "surgical": list(surgical), 
    "miscellaneous": list(miscellaneous)
}


os.makedirs("./subheading_categorization", exist_ok=True)
with open("./subheading_categorization/excluded_subheadings.json", "w", encoding="utf-8") as f:
    json.dump(excluded_categories, f)

excluded = set([i for item in excluded_categories.values() for i in item])
total = combine_sets(all_names) - excluded
combined = combine_and_filter(combined_names, exclude=excluded)

excluded_dict = {'category': [], 'heading_counts': []}
for k, v in excluded_categories.items():
    excluded_dict['category'].append(k)
    excluded_dict['heading_counts'].append(len(v))
df1 = pd.DataFrame(excluded_dict)

# Define all category names for easier access and modification
category_names = ["Maternal", "Vitals", "NEURO", "EENT", "CVS", "RESP", "GI", "GU", "Derm", "Endocrine", "MSK", "Lymphatic", "History"]

# Using dictionary comprehensions to fill categories
exclusive_categories = {cat.lower(): list(calculate_exclusive(cat, all_names, total)) for cat in category_names}

# Calculate intersected separately due to its unique calculation
intersected = total - combined
for cat in category_names:
    intersected -= calculate_exclusive(cat, all_names, total)

# Final categories dictionary
categories = {
    'all_inclusive': list(all),
    'excluded': list(excluded),
    'after_exclusion': list(total),
    'combined': list(combined),
    **exclusive_categories,
    'intersected': list(intersected)
}

with open("./subheading_categorization/subheading_categories.json", "w", encoding="utf-8") as f:
    json.dump(categories, f)

category_dict = {'category': [], 'heading_counts': []}
for k, v in categories.items():
    category_dict['category'].append(k)
    category_dict['heading_counts'].append(len(v))
df = pd.DataFrame(category_dict)

chart = alt.Chart(df).mark_bar().encode(
    x=alt.X('heading_counts:Q', title='Number of Headings'),
    y=alt.Y('category:N', title='Category', sort='-x')  # Sorting by descending order of counts
).properties(
    width=600,
    height=400,
    title='Distribution of Categorized Subheadings'
)

chart.display()
chart.save("Distribution_of_Categorized_Subheadings.pdf")

case_list = []
categories_set = set(categories["after_exclusion"])   
for case in new_case_list:
    new_case = {}
    filtered_items = {k: v for k, v in case["case"].items() if k in categories_set and v not in [None, "", [], {}]}
    title = case.get("title")
    
    if filtered_items and title:
        filtered_items["title"] = title
        case_list.append(filtered_items)

for case in case_list:
    for k, v in case.items():
        if k != "title":
            merged = ' '.join(v)
            case[k] = merged               

with open("./subheading_categorization/filtered_categories_case_list.json", "w") as f:
    json.dump(case_list, f)