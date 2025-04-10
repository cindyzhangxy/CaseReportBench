import json
import pandas as pd
import altair as alt
alt.data_transformers.enable("vegafusion")
import re

with open("./subheading_categorization/filtered_categories_case_list.json", "r") as f:
    case_list = json.load(f)

def space_normalizer(text):
    # Normalize different types of spaces and ensure regular spacing
    new_text = text.replace('\u2009', ' ').replace('\xa0', ' ')
    # Normalize space by replacing multiple spaces with a single space
    new_text = re.sub(r'\s+', ' ', new_text)
    return new_text.strip()

def count_words(text):
    words = text.split()
    return len(words)

def title_count_words(text):
    # Split text by any whitespace or hyphens
    words = re.split(r'[ -]+', text)
    return len(words)
    
new_case_list = []
for case in case_list:
    for k, v in case.items():

        if k != "title":  
            merged = ' '.join(v)
            if count_words(merged) > 3:
                case[k] = space_normalizer(merged)
        else:
            if case["title"]:
                case["title"] = case["title"]
    new_case_list.append(case)

for case in new_case_list:
    case["title"] = space_normalizer(case["title"].strip())

# Exclude case reports with titles that include treatment or adverse drug reactions 


tx_title = [report["title"] for report in new_case_list if "treat" in report["title"].lower() 
            or "therap" in report["title"].lower() 
            or "surg" in report["title"].lower() 
            and "diagno" not in report["title"].lower() ]

adr_title = [report["title"] for report in new_case_list if 
             "poison" in report["title"].lower() 
             or "adverse" in report["title"].lower() 
             or "side effect" in report["title"].lower()
             or "hypersensiti" in report["title"].lower()
             and "diagno" not in report["title"].lower() ]

excluded_titles = {"treatment_surgical": tx_title,
                   "poisoning_ADR": adr_title 
                  }
with open("./excluded_titles.json", "w" ) as f:
    json.dump(excluded_titles, f)

reduced_case_list = [report for report in new_case_list if report["title"] not in tx_title and report["title"]  not in adr_title]

# title that are less than ore  equals to 2 words
short_title = set([report["title"] for report in reduced_case_list  if title_count_words(report["title"]) <= 2 and "syndrome" not in report["title"].lower() ])

# Manually 
excluded_short_title = ['', '"', '100% N', '2+0', '46,XX,', '5′', 'A', 'A Fatal', 'A Homozygous', 'A New', 'A Novel', 'A Pathogenic', 'A Rare', 'A Recurrent', 'A monoallelic', 'A new', 'A nonsense', 'A novel', 'A pathogenic', 'A rare', 'A severe', 'ABCA1 Deficiency', 'AIDS-related', 'Accurate', 'Acquired', 'Acquired L718V/', 'Acute', 'Aggressive', 'Aggressive CD8', 'Airtraq', 'Aleatory', 'All-', 'Alterations in', 'An', 'An inferior', 'An insidious', 'An uninformative', 'Analysis of', 'Angioinvasive', 'Anomalous V', 'Anti Kp', 'Anti-', 'Anti-PLA', 'Apparent', 'Artificial', 'Association of', 'Autologous', 'Autotransplantation', 'Bacteriologically Determined', 'Beware of', 'Bi-allelic', 'Biallelic', 'Bifid sternum', 'Bilateral', 'Bilateral endogenous', 'Biological post', 'Biopsy-proven', 'Blueberry (', 'C-MAC', 'CD20', 'CD30', 'CD56', 'CD8', 'CO', 'CONSUMPTION OF', 'CPX‐351 (Vyxeos', 'CT and', 'Can', 'Canine non‐epitheliotropic', 'Carbapenem-Resistant', 'Carbapenemase producing', 'Carbapenemase-producing', 'Carcinosarcoma', 'Cardiac', 'Cardiac Amyloidosis', 'Cardiac lipoma', 'Cardiomyopathy Following', 'Carotidynia', 'Case Report:', 'Case of', 'Case report', 'Case report:', 'Case series:', 'Cat-induced', 'Catheter-Associated', 'Catheter-Related', 'Cefadroxil-Induced', 'Cefiderocol for', 'Cell‐free', 'Cerebral', 'Childhood', 'Chronic', 'Chylous Ascites', 'Cinematic', 'Classical', 'Cleft rhinoplasty', 'Clinically Infrequent', 'Clonally‐related primary', 'Clostridium', 'Co-Infection', 'Coexistence of', 'Coexisting of', 'Coinfection by', 'Coinfection of', 'Coinfection with', 'Coinheritance of', 'Colchicine Toxicity', 'Cold agglutinin', 'Combined', 'Commotio Cordis', 'Community acquired', 'Community-acquired', 'Community‐acquired', 'Comparing', 'Concomitant', 'Concurrent', 'Confirmation that', 'Contrast-enhanced', 'Coxsackie B', 'Co‐infection of', 'Co‐occurring', 'Crizotinib-Resistant', 'Cryobiopsy for', 'Cutaneous', 'Cutaneous CD56', 'Cytomegalovirus and', 'DORMEX', 'Dacryocystitis Involving', 'Deep cutaneous', 'Delayed onset', 'Delayed-onset', 'Dermatitis artefacta', 'Dermatomyofibroma', 'Dermatosis neglecta', 'Detecting', 'Detection of', 'Development of', 'Device-related', 'Diagnosing', 'Diagnosis of', 'Different', 'Diffuse Intense', 'Diffuse large', 'Diffusely increased', 'Diode Laser', 'Direct intralesional', 'Disseminated', 'Disseminated Invasive', 'Disseminated cutaneous', 'Disseminated focal', 'Distal', 'Disulfiram-Induced', 'Drug-free', 'Dual', 'Dual phase', 'Déjà Vu', 'EBV', 'Early-onset', 'EchoNavigator', 'Ectopic', 'Ectopic Human', 'Eculizumab-Associated', 'Effect of', 'Efficacy of', 'Emergence of', 'Emergency', 'Endobronchial', 'Endocarditis with', 'Endodontic implants', 'Endogenous', 'Entrectinib for', 'Epileptic Angina', 'Eradication of', 'Esophageal', 'Eumycetoma by', 'Evaluation of', 'Exflagellation of', 'Experience of', 'Expression of', 'Extensive', 'Extra-osseous', 'Extraintestinal', 'Extraosseous', 'Extremely low', 'F-FDG', 'Facial fistula', 'False Negative', 'False positive', 'False-positive', 'Fatal', 'Fatal Recurrent', 'Fatal necrotizing', 'Father–Son', 'Feline-transmitted', 'Fingerprinting MINOCA', 'Fluorescence', 'Folliculotropic CD8', 'Follow-up', 'Fractional CO', 'Fractional Co', 'From “', 'Fulminant', 'Fused', 'Gastric', 'Genetically Confirmed', 'Genome‐wide', 'Germline', 'Ginkgo Biloba', 'Glanders (', 'HER2/', 'HIV-related', 'Haemoglobin A', 'Heavy', 'Hemoglobin S/O', 'Hepatic', 'Hepatitis', 'Heterozygous', 'Heterozygous c.', 'High', 'Histamine H', 'History of', 'Homozygous missense', 'Huge', 'Hyper', 'Hyperinfection of', 'Hyperinfection with', 'Hyperkalemia by', 'Hypermucoviscous', 'Hypervirulent', 'Iatrogenic', 'Identification of', 'Identifying the', 'Idiopathic CD4', 'IgA-dominant', 'Imaging of', 'Immunoglobulin', 'Impact of', 'Impella RP', 'Importance of', 'Imported', 'Improvement of', 'Incidental', 'Infantile', 'Infection of', 'Infectious', 'Infratentorial', 'Integra', 'Integration of', 'Intense', 'Interictal', 'Intestinal', 'Intracoronary Lithotripsy', 'Intracranial', 'Intracranial fungal', 'Intraepithelial', 'Intrahepatic ovulation', 'Intraoperative transesophageal', 'Intravenous', 'Intraventricular adult', 'Invasive', 'Is', 'Isolated', 'Isolated Endobronchial', 'Isolated cervical', 'Isolated cutaneous', 'Isolated endogenous', 'Isolation of', 'Japanese', 'Karapandzic flap', 'Keratitis by', 'Keratitis with', 'Kushiyaki‐related', 'LATE ONSET', 'Laparoscopic repair', 'Largest', 'Larvae of', 'Late', 'Late-Onset', 'Lead Poisoning', 'Lethal', 'Linear', 'Linezolid-resistant', 'Live', 'Localized', 'Long', 'Lorlatinib for', 'Lumbar', 'Lung', 'Macro-B', 'Management of', 'Marked', 'Massive TAVR', 'Mature CD8', 'Melanoma', 'Melanosis oculi', 'Meningitis by', 'Meningitis for', 'Meropenem resistant', 'Mesenteric panniculitis', 'Metastatic chordoma', 'Metformin allergy', 'Methicillin resistant', 'Methicillin-Resistant', 'Methicillin-resistant', 'Methicillin‐resistant', 'Methidathion Poisoning', 'Misdiagnosis of', 'MitraClip', 'Mixed', 'Mosaic', 'Motor recovery', 'Mucinous Nevus', 'Multi-foci', 'Multi-system', 'Multicentric Ca', 'Multidrug Resistant', 'Multidrug resistant', 'Multidrug-Resistant', 'Multidrug-resistance', 'Multidrug-resistant', 'Multiorgan', 'Multi‐exon', 'Mutant', 'Myocarditis on', 'Myoepithelioma', 'Mysterious', 'Narcolepsy following', 'Native valve', 'Naturally acquired', 'Necrotizing', 'Negative', 'Neonatal', 'Neonatal invasive', 'New', 'Nexavar', 'Nifekalant', 'Nocardia', 'Non-Pathogenic', 'Non-serogroupable', 'Non-typhoid', 'Non-typhoidal', 'Non-zoonotic', 'Nonencapsulated', 'Nontyphoidal', 'Non‐pneumophila', 'Non‐pulmonary', 'Nosocomial', 'Novel', 'Novel germline', 'Novel heterozygous', 'Novel homozygous', 'Novel intergenic', 'OCT', 'OLOGEN', 'Occupationally acquired', 'Occurrence of', 'Ocular', 'Ocular Sporotrichosis', 'Ocular argyrosis', 'Ocular filariasis', 'Oculodentodigital dysplasia', 'Oesophageal intramural', 'Olanzapine induced', 'Onychocytic matricoma', 'Onychomycosis by', 'Opana', 'Organophosphate retinopathy', 'Orofacial granulomatosis', 'Otogenic', 'Outbreak of', 'Outpatient', 'Oxygen insufflation', 'P190', 'PaO', 'Pacemaker Associated', 'Palatogram revisited', 'Paradoxical embolism', 'Paradoxical hypertension', 'Paraneoplastic', 'Parasitism by', 'Parotid angiofibroma', 'Paroxetine‐induced', 'Patients harboring', 'PbtO', 'Pectus excavatum', 'Pediatric', 'Pediatric endogenous', 'Pellagra: A', 'Perioral', 'Periorbital', 'Peritoneal', 'Peritonitis by', 'Peritonitis with', 'Persistent', 'Pilomatricoma on', 'Placement of', 'Pleural', 'Polydactylyof 5', 'Poromas with', 'Positive', 'Possible', 'Post', 'Post Liposuction', 'Post cataract', 'Post-', 'Postoperative', 'Postpartum Endogenous', 'Post‐', "Potter's Sequence", 'Precipitation of', 'Prenatal‐onset', 'Preoperative', 'Presumptive complicating', 'Primary', 'Primary Cutaneous', 'Primary cardiac', 'Primary cutaneous', 'Primary pelvic', 'Probable', 'Probenecid', 'Probiotic related', 'Prolonged', 'Pulmonary', 'Pyonephrosis by', 'Quadriplegia after', 'Quinolone-resistant', 'R1352Q', 'Radioiodine (', 'Rapid', 'Rapidly progressive', 'Rare', 'Real-time', 'Recalcitrant', 'Reconstruction of', 'Recurrence of', 'Recurrent', 'Recurring', 'Reduction of', 'Refractory', 'Relapsed', 'Relapsing', 'Repeated', 'Report of', 'Resolution of', 'Resolving the', 'Response of', 'Responses to', 'Resuscitative', 'Return of', 'Revisiting Endosulfan', 'Rh E', 'Rheocarna', 'Risk of', 'Role of', 'Runaway pacemaker', 'Rupture of', 'SPECT/CT with', 'Sarcoidosis following', 'Seatbelt', 'Segmentation of', 'Selection of', 'Seronegative', 'Severe', 'Severe Disseminated', 'Severe Pulmonary', 'Severe invasive', 'Siblings With', 'Siddha', 'Simultaneous', 'Sodium', 'Solitary', 'Somatic', 'Splenectomy for', 'Spontaneous', 'Steroid and', 'Subacute', 'Subconjuctival', 'Subcutaneous', 'Subcutaneous Surprise', 'Subcutaneous entomophthoromycoses', 'Subcutaneous intralesional', 'Submental intubation', 'Success of', 'Successful', 'Successful Angiojet', 'Supratentorial', 'Survival from', 'Suspected', 'Symptomatic', 'Syndactyly', 'T2Candida', 'TTF1-positive', 'Tangled', 'The', 'The First', 'The Wandering', 'The invasive', 'The novel', 'Thoracic', 'Three Living', 'Three Novel', 'Three‐dimensional (', 'Tombs of', 'Toothpick meningitis', 'Toxoplasma Neuroretinitis', 'Tracheobronchopathia osteochondroplastica', 'Trails on', 'Transendocardial CD34', 'Transesophageal Echocardiography–Guided', 'Transformation of', 'Transient', 'Transoral CO', 'Transoral‐transpharyngeal CO', 'Transurethral', 'Transvaginal ureteroneocystostomy', 'Trigeminal schwannoma', 'Trigger Wrist', 'Two', 'Two novel', 'Ulceronecrotic Penicillosis', 'Ultrasound and', 'Underreporting', 'Under‐deployed', 'Unfortunately Fortunate', 'Unique', 'Unmasking Granulomatous', 'Unusual', 'Uptake of', 'Uremic stomatitis', 'Ureteroduodenal fistula', 'Urinary ascites', 'Urinary β', 'Use of', 'Usefulness of', 'Uterine', 'Utility of', 'Utilization behavior', 'Utilizing', 'Vaginal leiomyoma', 'Vancomycin-resistant', 'Variable urine', 'Variants in', 'Variants of', 'Vemurafenib for', 'Virulent', 'What', 'When', 'When amplified', 'Wide', 'Wintersweet (', 'Wrong incisions', 'Xanthomatous pleuritis', 'X‐linked', 'Zoon vulvitis', 'p16', 'p210', 't(9;14)(p13;q32)/', '‘When', '“', '＜Editors’ Choice＞']


case_list = [report for report in reduced_case_list if report["title"] not in excluded_short_title]

with open ("./refined_case_list.json", "w") as f:
    json.dump(case_list, f)