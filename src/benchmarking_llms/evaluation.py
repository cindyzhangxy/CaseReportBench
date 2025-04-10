from datasets import load_dataset, load_from_disk
import pandas as pd
import json
from preprocessing_llm_output import *
from collections import defaultdict
import numpy as np
from eval_metrics import *
import evaluate
from prettytable import PrettyTable, TableStyle
import math
 
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

# def count_missing_k(cell):
#     if isinstance(cell, list):  # Check if the cell is a list
#         return cell.count("missing_k")  # Count "missing_k" in the list
#     return 0  # Return 0 if the cell is not a list

def read_table_from_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

        table = PrettyTable()
        table.field_names = data["columns"]

        for row in data["rows"]:
            table.add_row(row)
        return table

def save_table_as_json(table, filename):
    data = {
        "columns": table.field_names,
        'rows': table._rows
    }

    with open(f'./{filename}', 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def calculate_bleu_column_wise(df_pred, df_ref):
    """
    Calculates BLEU and ROUGE scores column-wise for two pandas DataFrames.

    This function computes BLEU and ROUGE scores for each cell in the DataFrame, 
    where `df_pred` contains the predicted outputs and `df_ref` contains the 
    reference outputs. It gracefully handles edge cases, such as when either 
    the reference or prediction is empty.

    Special Cases:
    - If both `translation_length` and `reference_length` are 0, assigns a perfect 
      BLEU score (BLEU = 1.0, precisions = [1.0, 1.0, 1.0, 1.0]).
    - If either `translation_length` or `reference_length` is 0 but not both, 
      assigns a BLEU score of 0.0 (precisions = [0.0, 0.0, 0.0, 0.0]).
    - ROUGE scores are calculated separately and assumed to handle empty inputs gracefully.
    """
    df_pred = df_pred.map(lambda x: ["unk"] if isinstance(x, list) and not x else x)
    df_ref = df_pred.map(lambda x: ["unk"] if isinstance(x, list) and not x else x)

    assert df_pred.shape == df_ref.shape, "Aligned DataFrames must have the same shape."

    bleu_scores = {}
    rouge_scores = {}

    for col in df_pred.columns:
        column_bleu = []
        column_rouge = []

        for idx in df_pred.index:
            pred = " ".join(df_pred.at[idx, col]) if isinstance(df_pred.at[idx, col], list) else df_pred.at[idx, col]
            ref = " ".join(df_ref.at[idx, col]) if isinstance(df_ref.at[idx, col], list) else df_ref.at[idx, col]

            translation_length = len(pred)
            reference_length = len(ref)

            if translation_length == 0 and reference_length == 0:
                bleu_score = {
                    "bleu": 1.0,
                    "precisions": [1.0, 1.0, 1.0, 1.0],  # Assign perfect precision
                    "translation_length": translation_length,
                    "reference_length": reference_length
                }
            elif translation_length == 0 or reference_length == 0:
                bleu_score = {
                    "bleu": 0.0,
                    "precisions": [0.0, 0.0, 0.0, 0.0],  # Assign zero precision
                    "translation_length": translation_length,
                    "reference_length": reference_length
                }
            else:
                try:
                    bleu_score = bleu_metric.compute(predictions=[pred], references=[[ref]])
                    bleu_score.update({
                        "translation_length": translation_length,
                        "reference_length": reference_length
                    })
                except ZeroDivisionError:
                    bleu_score = {
                        "bleu": 0.0,
                        "precisions": [0.0, 0.0, 0.0, 0.0],
                        "translation_length": translation_length,
                        "reference_length": reference_length
                    }

            column_bleu.append(bleu_score)
            # ROUGE is assumed to handle zero lengths gracefully
            rouge_score = rouge_metric.compute(predictions=[pred], references=[[ref]])
            column_rouge.append(rouge_score)

        bleu_scores[col] = column_bleu
        rouge_scores[col] = column_rouge

    return bleu_scores, rouge_scores
    
def check_for_omission_hallucination_category_wise(df1, df2, name1, name2):
    """
    Computes omission and hallucination percentages for each column (category) in the DataFrames.
    - df1: LLM predictions
    - df2: Reference annotations
    - name1, name2: Names for debugging/logging purposes
    """
    # Align DataFrames to ensure they have the same columns
    df1, df2 = df1.align(df2, join="outer", axis=1)

    omission_hallucination_scores = {}

    for col in df1.columns:
        # Check for empty lists in both columns
        llm_empty = df1[col].map(lambda x: x == [])
        ref_empty = df2[col].map(lambda x: x == [])

        # Compute Omission (Reference has information but LLM extracts nothing)
        ref_has_info = ~ref_empty  # Cases where reference has content
        omission_cases = ref_has_info & llm_empty  # LLM failed to extract but reference had content
        total_ref_cases = ref_has_info.sum()  # Total cases where reference has content
        
        omission_percentage = (omission_cases.sum() / total_ref_cases * 100) if total_ref_cases > 0 else 0

        # Compute Hallucination (LLM extracts information where reference has none)
        hallucination_cases = ~ref_has_info & ~llm_empty  # LLM extracted when reference had nothing
        total_no_ref_cases = (~ref_has_info).sum()  # Cases where reference was empty

        hallucination_percentage = (hallucination_cases.sum() / total_no_ref_cases * 100) if total_no_ref_cases > 0 else 0

        # Store results
        omission_hallucination_scores[col] = {
            "Omission (%)": omission_percentage,
            "Hallucination (%)": hallucination_percentage
        }

    return omission_hallucination_scores


def get_metrics(llm_output_path, zsCombined=False):
    evaluation_order = [ "Vitals_Hema","Neuro", "EENT", "CVS", "RESP", "GI", "GU", "MSK", "DERM", "Lab_Image",  "LYMPH", "History", "ENDO", "Pregnancy"]
    # Load Annotator Ground Truth
    gt_df = pd.read_json("../result/gt_annotator_df.json", orient="records")
    gt_df = gt_df.map(lambda x: [] if x is None or (isinstance(x, float) and pd.isna(x)) else x)
    gt_df["pmcid"].astype(int)
    gt_df.sort_index(inplace=True, kind="stable")
    assert gt_df.isna().sum().sum()==0,  f"There are NaN values in the DataFrame"
  

    with open(llm_output_path, "r", encoding="utf-8") as f:
        llm_output = json.load(f)

    if zsCombined:
        finalized_output = zsCombined_standardized_finalized_llm_output(llm_output, gt_df)
                
    finalized_output = standardized_finalized_llm_output(llm_output)

    
    llm_df = make_df(finalized_output)
    llm_df = llm_df.map(process_item)
    llm_df["pmcid"].astype(int)
    assert llm_df.isna().sum().sum() == 0, f"There are NaN values in the DataFrame:\n{llm_df[llm_df.isna().any(axis=1)]}"
 
    
    gt_df.set_index("pmcid", inplace=True)
    llm_df.set_index("pmcid", inplace=True)
    
    llm_df.sort_index(inplace=True, kind="stable")
    gt_df.sort_index(inplace=True, kind="stable")


    gt_df = gt_df[evaluation_order]
    llm_df = llm_df[evaluation_order]

    assert gt_df.columns.equals(llm_df.columns), "Columns of gt_df and llm_df do not match"
    assert gt_df.index.equals(llm_df.index), "Indices of gt_df and llm_df do not match"

    
    # percent_missing_k = df.applymap(count_missing_k).sum()/len(df)*100

    omission_category_wise = check_for_omission_hallucination_category_wise(llm_df, gt_df, name1="llm_df", name2="gt_df")
    hallucination_category_wise = check_for_omission_hallucination_category_wise(gt_df, llm_df, name1="gt_df", name2="llm_df")
 

    # To perform token set ratio 
    scores_df = calculate_token_set_ratio_scores_ray(llm_df, gt_df)
    mean_token_set_ratio = scores_df.mean().mean()
    category_wise_mean_token_set_ratio_df = scores_df.mean(axis=0)

    bleu, rouge = calculate_bleu_column_wise(df_pred=llm_df, df_ref=gt_df)

    bleu_1_avg = {}
    for k, entries in bleu.items():
        bleu_1_avg[k] = sum(entry['precisions'][0] for entry in entries) / len(entries)
    bleu_4_avg = {}
    for k, entries in bleu.items():
        bleu_4_avg[k] = sum(entry['precisions'][3] for entry in entries) / len(entries)
    rougeL_avg = {}
    for k, entries in rouge.items():
        rougeL_avg[k] = sum(entry['rougeL'] for entry in entries) / len(entries)

    results_df = pd.DataFrame({
        'Category': list(bleu_1_avg.keys()),
        'BLEU-1 Avg': [round(bleu_1_avg[cat], 4) for cat in bleu_1_avg.keys()],
        'BLEU-4 Avg': [round(bleu_4_avg[cat], 4) for cat in bleu_4_avg.keys()],
        'ROUGE-L Avg': [round(rougeL_avg[cat], 4) for cat in rougeL_avg.keys()],
        'Mean Token Set Ratio': [round(category_wise_mean_token_set_ratio_df[cat], 4) for cat incategory_wise_mean_token_set_ratio_df.keys()],
        'Omission (%)': [round(omission_category_wise.get(cat, 0), 2) for cat in bleu_1_avg.keys()],
        'Hallucination (%)': [round(hallucination_category_wise.get(cat, 0), 2) for cat in bleu_1_avg.keys()]
    })

    # Add average metrics to the DataFrame
    results_df = pd.concat([
        results_df,
        pd.DataFrame({
            'Category': ['Average'],
            'BLEU-1 Avg': [round(np.mean(list(bleu_1_avg.values())), 4)],
            'BLEU-4 Avg': [round(np.mean(list(bleu_4_avg.values())), 4)],
            'ROUGE-L Avg': [round(np.mean(list(rougeL_avg.values())), 4)],
            'Mean Token Set Ratio': [round(category_wise_mean_token_set_ratio_df.mean(), 4)],
            'Omission (%)': [round(np.mean(list(omission_category_wise.values())), 2)],
            'Hallucination (%)': [round(np.mean(list(hallucination_category_wise.values())), 2)]
        })
    ])

    # Create PrettyTable
    # table = PrettyTable()
    # table.field_names = ['Category', 'BLEU-1 Avg', 'BLEU-4 Avg', 'ROUGE-L Avg', 'Mean Token Set Ratio', 'Omission (%)', 'Hallucination (%)']
    
    # for _, row in results_df.iterrows():
    #     table.add_row(row.tolist())

    # table.align = 'c'
    # table.align['Category'] = 'l'
    # table.set_style(TableStyle.DOUBLE_BORDER)

    # return table, results_df

    return results_df