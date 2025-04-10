import click
import json
import os
from extractAug24 import *

def get_report(report):
    """
    Process the entire report using SYSTEMIC_EXTRACTION.
    Concatenate all text fields into a single string and apply simple_extraction.
    """
    # Combine all fields into a single string
    full_text = " ".join([str(value) for value in report.values()])

    # Apply simple_extraction with SYSTEMIC_EXTRACTION
    try:
        extracted_data = simple_extraction(full_text, SYSTEMIC_EXTRACTION)
        return extracted_data
    except Exception as e:
        print(f"Error processing report: {e}")
        return {"error": str(e)}

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
def main(input_file):
    """
    Process the input JSON file and extract structured information using SYSTEMIC_EXTRACTION.
    """
    # Load the input JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        try:
            case_list = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading JSON: {e}")
            return

    # Ensure input is a list of dictionaries
    if not isinstance(case_list, list) or not all(isinstance(report, dict) for report in case_list):
        print("Input JSON must be a list of dictionaries, where each dictionary represents a report.")
        return

    # Process each report
    output = []
    for i, report in enumerate(case_list):
        print(f"Processing report {i + 1}/{len(case_list)}...")
        result = get_report(report)
        output.append(result)

    # Save the output to a file
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