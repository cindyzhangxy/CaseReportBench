import click
import json
import os
from datasets import load_dataset
from extractAug24 import *

# Load category data
with open('/scratch/st-jzhu71-1/czhang/job/subheading_categorization/subheading_categories.json', "r", encoding="utf-8") as f:
    categories = json.load(f)
# Define categories
general = set(categories['combined']).union(set(categories["intersected"]))
maternal = set(categories["maternal"])
neuro = set(categories['neuro'])
eent = set(categories['eent'])
cvs = set(categories['cvs'])
resp = set(categories['resp'])
gi = set(categories['gi'])
gu = set(categories['gu'])
derm = set(categories['derm'])
endo = set(categories['endocrine'])
msk = set(categories['msk'])
lymph = set(categories['lymphatic'])
history = set(categories['history'])

def extract_and_append(report_dict, key, values, extractor):
    """Helper function to apply extraction and append results."""
    if key not in report_dict:
        report_dict[key] = []
    extraction_result = simple_extraction(values, extractor)
    if extraction_result:  # Only append if the extraction yielded something
        report_dict[key].append(extraction_result)
        
def get_report(report):  
    
    report_dict = {}
    # title = report.get("title", "")   
    # if title:  # Only process the title if it is not empty
    #     report_dict['diagnosis'] = simple_extraction(title, Diagnosis)
    para = " ".join([str(l) for l in report.values()])
    report_dict["iem"] = simple_extraction(para, IEM)
    
    for key, values in report.items():
            
        if key in general:
            extract_and_append(report_dict, 'Vitals_Hema', values, Vitals_Hema)
            extract_and_append(report_dict, 'Pregnancy', values, Pregnancy)
            extract_and_append(report_dict, 'Neuro', values, Neuro)
            extract_and_append(report_dict, 'CVS', values, CVS)
            extract_and_append(report_dict, 'RESP', values, RESP)
            extract_and_append(report_dict, 'EENT', values, EENT)
            extract_and_append(report_dict, 'GI', values, GI)
            extract_and_append(report_dict, 'GU', values, GU)
            extract_and_append(report_dict, 'DERM', values, DERM)
            extract_and_append(report_dict, 'MSK', values, MSK)
            extract_and_append(report_dict, 'ENDO', values, ENDO)
            extract_and_append(report_dict, 'LYMPH', values, LYMPH)
            extract_and_append(report_dict, 'History', values, History)
        elif key in maternal:
            extract_and_append(report_dict, 'Pregnancy', values, Pregnancy)
        elif key in neuro:
            extract_and_append(report_dict, 'Neuro', values, Neuro)
        elif key in cvs:
            extract_and_append(report_dict, 'CVS', values, CVS)
        elif key in resp:
            extract_and_append(report_dict, 'RESP', values, RESP)
        elif key in eent:
            extract_and_append(report_dict, 'EENT', values, EENT)
        elif key in gi:
            extract_and_append(report_dict, 'GI', values, GI)
        elif key in gu:
            extract_and_append(report_dict, 'GU', values, GU)
        elif key in derm:
            extract_and_append(report_dict, 'DERM', values, DERM)
        elif key in msk:
            extract_and_append(report_dict, 'MSK', values, MSK)
        elif key in endo:
            extract_and_append(report_dict, 'ENDO', values, ENDO)
        elif key in lymph:
            extract_and_append(report_dict, 'LYMPH', values, LYMPH)
        elif key in history:
            extract_and_append(report_dict, 'History', values, History)

    return report_dict


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
def main(input_file):
    """Process the input JSON file and perform operations defined in get_report."""
    with open(input_file, "r", encoding="utf-8") as f:
        case_list = json.load(f)

    output = [get_report(report) for report in case_list]

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_directory = "./extracted_output"
    output_filename = f"{base_name}_out.json"
    output_path = os.path.join(output_directory, output_filename)

    os.makedirs(output_directory, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as g:
        json.dump(output, g, ensure_ascii=False, indent=4)
    print(f"Output written to {output_path}")

if __name__ == "__main__":
    main()
