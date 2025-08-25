import os

import zipfile

import xml.etree.ElementTree as ET

from docx import Document as DocxDocument

from openpyxl import Workbook

from crewal import Agent, Task, Crew, LLM

from langgraph.graph import StateGraph

from openai import AzureOpenAI

import httpx

from dotenv import load dotenv

from msal import ConfidentialClientApplication

import json

from typing import List, Dict, Tuple

from datetime import datetime

import httpx

import pandas as pd

import json

import re

from langchain.schema import Document

import base64

load_dotenv()

import truststore

truststore.inject_into_ssl()

#Access the variables

OPENAI AZURE CLIENT_ID os.getenv("OPENAI_AZURE_CLIENT_ID ")

OPENAI AZURE CLIENT_SECRET os.getenv("OPENAI_AZURE_CLIENT_SECRET")

OPENAI_AZURE TENANT_ID os.getenv("OPENAI_AZURE_TENANT_ID")

ΟΡΕΝΑΙ ΑΡΙ ΤTYPE "azure_ad"

OPENAI API VERSION "2023-05-15"

OPENAI GPT_DEPLOYMENT_NAME = "gpt4omni_prod_sec_2024-08-06"

OPENAI EMBEDDING_DEPLOYMENT_NAME = "ada002_2"

client AzureOpenAI(

api_key-openai_api_key,

) api_version-OPENAI API VERSION, azure_endpoint-OPENAI_CHAT_API BASE, azure_deployment OPENAI_GPT_DEPLOYMENT_NAME,

def create 11m():

"Create an Azure OpenAI client using the access token.""**

access_token= get_access_token()

chat_api_base os.getenv("AZURE_CHAT_API_BASE") + os.getenv("AZURE_CHAT_API_SUFFIX")

os.environ['NO_PROXY]-".azure-api.net"

os.environ["OPΕΝΑΙ_ΑΡΙ_ΚΕΥ"] = access_token

os.environ ["AZURE OPENAI API KEY"] access token

os.environ["AZURE_OPENAI_ENDPOINT"] os.environ["AZURE OPEΝΑΙ ΑPI_VERSION"] chat api_base "2023-05-15"

11m LLM( model-f'azure/{os.getenv("AZURE_CHATGPT_DEPLOYMENT_NAME")}', api_key-access_token, base_url-chat_api_base, api version="2023-05-15", temperature-0.3,

) #print('11mmmmmmmmmmmmmmmmmmmmmmm, 11m)

I

return 11m

def call_azure_openai(prompt):

response client.chat.completions.create( ) model-"gpt-40", messages-[["role": "user", "content": prompt)]

return response.choices[0].message.content

#File Readers

def extract_text_from_docx(docx_path, image_output_dir="extracted_images"):

#Extract text

doc DocxDocument (docx_path)

text "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

#Extract images

image files[]

with zipfile.ZipFile(docx_path, 'r') as docx_zip:

image files [f for f in docx_zip.namelist() if f.startswith("word/media/")]

os.makedirs(image_output_dir, exist_ok-True)

for image file in image files:

image pathos.path.join(image_output_dir, os.path.basename(image_file))

with open (image_path, "wb") as img put:

img_out.write(docx_zip.read(image_file))

#Create LangChain Document

metadata = {

"source": docx_path,

"images": [os.path.join(image_output_dir, os.path.basename(img)) for img in image_files]

return [Document(page_content text, metadata metadata)]

def extract_text_from_excel(excel_path, sample_rows-20): xls = pd.ExcelFile(excel_path, engine='openpyxl*)

sheet_name = xls.sheet_names [0]

df = pd.read_excel(xls, sheet_name=sheet_name)

records df.to_dict(orient="records")

return {

}

"sheet_name": sheet_name,

"rows": len(df),

"columns": df.shape[1],

"records": records [:50],

"file_type": "excel"

def extract logic_from_twbx(twbx_path):

extracted text ---

with zipfile.ZipFile(twbx path, 'r') as zip_ref: zip_ref.extractall("extracted_twbx")

twb file None

for file in os.listdir("extracted_twbx"):

if file.endswith(".twb"):

twb_file os.path.join("extracted_twbx", file)

break

if not twb file:

return "No.twb file found in twbx archive."

tree ET.parse(twb_file)

root tree.getroot()

for calc in root.findall(".//calculation"):

extracted_text + f"Calculation: (calc.attrib.get('formula', '')}\n"

for filter_elem in root.findall(".//filter"):

extracted_text + f"Filter: (ET.tostring(filter elem, encoding'unicode")}\n"

for annotation in root.findall(".//annotation"):

extracted_text += f"Annotation: (ET.tostring(annotation, encoding-unicode')}\n"

return extracted text

def encode_images_to_base64(Image_paths):

encoded_images = []

for path in image_paths:

with open(path, "rb") as img file encoded base64,b64encode(img_file.read()).decode("utf-8")

encoded_images.append({"filename": os.path.basename(path), "base64": encoded))

return encoded_images

def extract_text_from_csv(csv_path, sample_rows-10):

dfpd.read_csv(csv_path)

return df.to dict(orient="records") [:50]

fsd_reader Agent(

name "FSD Reader",

role="Functional Specification Analyst",

goal-"Extract detailed business rules, filter logic, report derivation, and validation rules from FSD documents",

backstory-"Expert in interpreting functional specs including dropdown behavior, default values, SQL mappings, and drill-down logic",

Ilm-create_ll1m(),

verbose-True

#allow_delegation-False

report_analyst - Agent(

name "Tableau Report Analyst",

role-"BI Dashboard Analyst",

goal-"Extract KPIs, calculated fields, filters, and data sources from Tableau reports",

backstory"Skilled in reverse-engineering Tableau dashboards to understand business logic and metrics",

11m-create_11m(),

verbose-True

allow delegation-False

regression_analyst - Agent(

name "Regression Test Case Analyst",

role"Regression Test Case Expert",

goal-"Analyze regression test cases to extract navigation steps, SQL validations, filter logic, and edge case scenarios",

backstory="Experienced in identifying reusable logic, SQL validation patterns, and UI-to-backend traceability from historical test cases",

11m-create_llm(),

verbose-True

#allow delegation-False

test_designer = Agent(

)

name="Test Case Generator",

role="QA Automation Specialist",

goal-"Design comprehensive test cases based on business rules and report logic, covering edge cases and validations",

backstory="Experienced in translating business logic into robust test scenarios for data validation and reporting accurad

11m-create_l1m(),

verbose-True

#allow_delegation=False

qa_reviewer = Agent(

)

name="QA Reviewer",

role="Senior QA Lead",

goal-"Review test cases for completeness, clarity, and alignment with business requirements",

backstory="Ensures that test cases meet quality standards and cover all functional and reporting aspects",

11m-create_l1m(),

verbose-True

# allow_delegation=False

disclosure_mapper Agent(

name="Disclosure Mapper",

role="Disclosure Mapping Analyst",

goal="Extract and interpret disclosure mappings and FCCS classifications from derivative screen",

backstory="Expert in financial disclosure structures and mapping logic used in regulatory reporting",

11m-create_1l1m(),

verbose True

# allow_delegation-False

)


def fsd node(state):

LangGraph node function that processes FSD documents using CrewAl agent

#Extract text from the FSD document

documents extract_text_from_docx(state["fsd_path"])

fsd_doc documents[0]

fsd text fsd_doc.page_content

image paths fsd_doc.metadata.get("images", [])

encoded_images encode_images_to_base64(image_paths)

#Format image info for the agent

image_info "\n"-join([

f"[img['filename']): (base64 content truncated)\n{img['base64"]}..." for ing in encoded_images

]) if encoded_images else "No images found."

fsd_analysis_task - Task(

description-f"""

Analyze the EFRA Reporting Layer Functional Specification and extract:

1. Business rules and logic for filters, dropdowns, and default values

2. Derivation logic for report rows and columns

. SQL mappings and data source references (e.g., FCCS Disclosure Balances)

3 4. Hierarchy aggregation logic for Ledger Entity, Custom 3, Account

5. Drill-down behavior and navigation logic

6. Validation rules and constraints

7. Any other relevant logic for generating test cases and validating reports

Please provide a structured output categorized by:

Filters and Parameters

Derivation Logic (Rows, Columns)

Data Sources and SQL Mapping

Navigation and Drill-down Behavior

Validation Rules

FSD Content:
(fsd text)

Embedded Images:

{image_info)

agent-fsd_reader,

) expected_output "Structured analysis of EFRA FSD including filter logic, derivation rules, SQL mappings, and drill-down behavior

#Create a crew with the agent and task

fsd crew Crew(

)

agents-[fsd_reader],

tasks [fsd_analysis_task],

verbose-True, #Optional: for debugging

process-"sequential"

#Execute the crew and get results

try:

result fsd crew.kickoff()

rules result if isinstance(result, str) else str(result)

except Exception as e:

print("Error processing FSD with CrewAI: (e)")

rules f"Error processing FSD: (str(e))"

#Return updated state

return (**state, "fsd_rules": rules)




def tableau_node(state):

report folder state["report_folder"]

combined report text

for file name in os.listdir(report_folder):

if file_name.endswith(".twbx"):

file pathos.path.join(report_folder, file name) report text extract_logic_from_twbx(file_path)

combined_report_text += "\n-- Report: (file_name}-\n(report_text)\n"

#Get context from previous nodes

fsd_context state.get("fsd_rules", "")

regression_context state.get("regression_insights", "")

tableau task - Task(

description-f"""

Extract Key Performance Indicators, calculated fields, filters, and business logic from Tableau reports.

Context from FSD Analysis: (fsd_context)

Context from Regression Analysis: (regression_context)

Tableau Reports Content: (combined_report_text)

Extract and analyze:

1. Key Performance Indicators (KPIs) and metrics

2. Calculated fields and their business logic

3. Filters and parameter logic

4. Data source relationships and joins

5. Dashboard interactions and dependencies

6. Validation rules embedded in reports

Cross-reference with FSD business rules to ensure alignment and identify any discrepancies.
"""

agent-report_analyst,

) expected_output="Comprehensive analysis of Tableau report logic, KPIs, and business rules with cross-references to FSD"

crew Crew(

) agents=[report_analyst], tasks-[tableau_task], process-"sequential"

result crew.kickoff()

kpis result if isinstance(result, str) else str(result)

return {

**state,

"report_kpis": kpis

}

def test case_node(state):

fsd_rules state.get('fsd_rules", "")

report kpis state.get('report_kpis', '')

regression_insights state.get('regression_insights','')

disclosure_mappings state.get('disclosure_mappings", **)

efra_insights state.get('efra_insights',)

efra_t5_path state["efra_t5_path"]

efra_records extract_text_from_csv(efra_t5_path)

disclosure_path state["disclosure_path"]

disc_metadata extract_text_from_excel(disclosure_path)

disclosure_records disc_metadata["records"]

#Extract text from the FSD document

fsd documents extract_text_from_docx(state["fsd_path"])

fsd doc fsd_documents[0]

fsd text fsd_doc.page_content

image paths fsd_doc.metadata.get("images", [])

encoded_images encode_images_to_base64(image_paths)

test_generation_task Task(

Generate test cases using only the report names and disclosure views that are explicitly mentioned in the current FSD document provided below.

description-f"""

You must:

Extract valid report names and disclosure views from the FSD content.

Use only those names in test case titles, steps, and expected results.

Do not reuse names from regression insights unless they are also present in the FSD.

FSD Content:

{fsd_text)

Make sure that all names like Disclosure view, report names, and other references in test steps, test data, and Expected Result of all the test coses generated are strictly based on the current fsd doc and are absolutely correct: (fsd_text)

Requirements:

1. Prioritize business rules and dropdown logic from FSD and EFRA

2. Include navigation steps for Tableau and Athena

3. Incorporate SQL queries for backend validation using EFRA records: (efra_records)

4. Cover positive, negative, and edge cases based on current logic

5. Include drill-down behavior and filter carry-forward logic from Disclosure mappings: (disclosure_records)

6. Must have validation steps at the end of each test case

7. Ensure test cases are executable by a novice

Structure each test case with:

Test Case ID and Name

Objective/Purpose

Pre-conditions

Test Steps (with navigation)

Expected Results against each Test step

Validation step for each Test

Test Data Requirements

Priority Level

agent-test designer,

expected output "Structured test cases with navigation, SQL validation, and coverage of all business logic. Also make sure to include Navigation for each and every step such that it should be detailed enough for a novice to execute without any external help. Give complete SQL queries everywhere so as the user will just execute them."

crew Crew(agents-[test_designer], tasks-[test_generation_task), process-"sequential")

result crew.kickoff()

test cases result if isinstance(result, str) else str(result)

return (**state, "test cases"; test_cases)





def

review_node(state):

fsd_rules state.get('fsd_rules', '")

report kpis state.get('report_kpis', '')

regression_insights state.get('regression_insights', ') disclosure_mappings state.get('disclosure_mappings','')

efra_insightsstate.get('efra_insights','')

test_cases state.get('test_cases', '')

sample_df pd.read_excel(test_case_excel_path)

column_names list(sample_df.columns)

sample_rows sample_df.head(3).to_dict('records')

efra_t5 path state["efra_t5_path"]

efra_records extract_text_from_csv(efra_t5_path)

disclosure_path state["disclosure_path"]

disc metadata extract_text_from_excel(disclosure_path)

disclosure_records disc_metadata["records"]

#Extract text from the FSD document

fsd_documents = extract_text_from_docx(state["fsd_path"])

fsd_doc fsd_documents[0]

fsd text fsd_doc.page_content

review task Task(

description-f"""

Review and finalize test cases for completeness and quality based on context below.

Context:

- FSD Rules: {fsd_rules}

Tableau KPIs: (report_kpis)

Disclosure Mappings: (disclosure_mappings)

EFRA Insights: (efra_insights)

Test Cases: (test_cases]

Sample Format:

Columns: [column_names}

Sample Rows: (json.dumps(sample_rows, indent-2))

Instructions:

1. Ensure each test case aligns with business rules and regression logic

2. Validate SQL queries against EFRA mappings

3. Format each test case to match sample Excel structure and take them as sample only test generation procss should not be modified in any way because of that.

4. Output a 350N array with keys matching: {column_names)

5. Find and resolve any missing coverage or gaps

7. Incorporate SQL queries for backend validation, and make sure to modify all the sql queries generated to have correct disclosure id, columns, jains based on

6. Generate as many quality test cases as possible for 100 percent coverage or to cover gaps.

( efra_records)

8. Cover positive, negative, and edge cases

9. Make sure that the test steps, test data and Expected Result of all the generated test cases generated are strictly based on the current fod doc: (fsd test)

10. Include drill-down behavior and filter carry-forward logic: (disclosure_records)

11. Give complete SQL queries everywhere so as the user will just execute them

Review Criteria:

Completeness

SQL. Validation Accuracy and it's modified correctly based on the disclosure on which test cases are being generated

Navigation Clarity

Formatting Consistency

Coverage of Drill-down and Filter Logic

Must have Validation steps at the end of each Test case to verify results

Generate as many quality test cases as possible for 100 percent coverage or to cover gaps.

Complete SQL queries everywhere so as the user will just execute them

All the Test Steps, Test Data and Expected Result of all the generated test cases generated are strictly based on the current fsd_doc: (fsd text)

Structure test cases with:

Test Case ID and Name

Objective/Purpose

Pre-conditions

Test Steps (with navigation)

Expected Results against each Test step

Validation step for each Test

Test Data Requirements

Include Navigation for each and every step

Priority Level

agent-qa_reviewer,

=

) expected_output="ONLY JSON array of reviewed test cases formatted to match Excel structure"

crew Crew(agents=[qa_reviewer], tasks-[review_task], process-"sequential")

result crew.kickoff()

try:

cleaned_input re.sub(r'^json$', '', str(result).strip(), flags-re.MULTILINE)

data json.loads(cleaned_input)

output_df pd.DataFrame(data)

output_df.to_csv("./generated_test_cases_new.csv", index=False)

print(f"csv generated")

return {

}

**state,

"final_test_cases": result,

"test_cases_excel": output_path,

"test_cases_dataframe": output_df.to_dict('records')

except Exception as e:

return {

**state,

"final_test_cases"; result, "formatting_error": str(e)

}




def disclosure_node(state):

disclosure_path = state["disclosure_path"]

disc_metadata extract_text_from_excel(disclosure_path)

disclosure_records disc_metadata["records"]

disclosure_task = Task(

description-f"""

Analyze the derivative screen data and extract:

1. Disclosure categories and their FCCS mappings

2. Relationships between Custom fields (C1, C3, C4) and disclosure types

3. Any logic used to derive lines/columns in reporting

4. Any notes or formulas used in the sheet

=

Data of disclosure excel using which test cases are to generated for current report: {disclosure_records}

agent-disclosure_mapper,

) expected_output="Structured mapping of disclosures, FCCS classifications, and reporting logic"

crew Crew(agents [disclosure_mapper], tasks-[disclosure_task], verbose-True)

result crew.kickoff()

return (**state, "disclosure_mappings": result)




def efra_node(state):

efra_t5_path efra_records state["efra_t5_path"] extract_text_from_csv(efra_t5_path)

efra task Task(

description-f"""

Analyze EFRA T5 financial data and extract:

1. Entity-level financial patterns

2. Currency-specific reporting logic

3. High-value transactions and their implications

4. Any anomalies or edge cases worth testing

5. Useful SQL filters or joins based on custom dimensions

T5 Extract of data on which a tableau report was created and test cases to be generated: (efra_records)

agent-efra_analyst, expected_output-"Insights from EFRA T5 data to guide test case generation and validation. Also Useful SQL filters or joins or queries that can be used to generate test cases with appropriate columns, tables, disclosure id etc."

) crew Crew(agents-[efra_analyst], tasks-[efra_task], verbose True)

result crew.kickoff() return (**state, "efra_insights": result)


def regression_node(state):

regression path state["regression_path"]

disc_metadata extract_text_from_excel(regression_path)

disclosure_records disc metadata["records"]

#Get FSD rules from previous node for context

fsd_context state.get("fsd_rules", "")

regression_task - Task(

description-f

Analyze the regression test cases and extract:

1. Navigation steps and UI interactions

2. SQL queries used for backend validation

3. Filter logic and dropdown behavior

4. Color-coded threshold validations (green, red, blue)

5. Report title and layout validation rules

. Drill-down navigation and filter carry-forward logic

6 7. Reusable test patterns and edge cases

8. Any gaps or inconsistencies in coverage

Please structure the output into:

Navigation Steps

SQL Validation Patterns

Filter Logic

Threshold Rules

Report Layout Expectations

Drill-down Behavior

Reusable Test Templates

regression test cases on previous reports using which new test

cases are to be generated (they are to be used for reference only):

{disclosure_records)

agent-regression_analyst,

expected_output "Structured insights from regression test cases including

SQL validation, navigation, and layout rules"


crew = Crew(

) agents=[regression_analyst], tasks=[regression_task], process="sequential"

result = crew.kickoff()

insights result if isinstance (result, str) else str(result)

return {

**state,

"regression_insights": insights insights
}

#Excel Writer---

def write_test_cases_to_excel(test_cases_text, output_path="generated_test_cases.xlsx"):

wb Workbook()

WS wb.active

ws.title "Test Cases"

for i, line in enumerate(test_cases_text.splitlines(), start-1):

ws.cell(row-i, column-1, value-line)

wb.save(output_path)

return output_path

#LangGraph Workflow

graph StateGraph(dict)

#Existing nodes

graph.add_node("fsd", fsd_node)

graph.add_node("regression", regression_node)

graph.add_node("report", tableau_node)

#New nodes

graph.add_node("disclosure", disclosure_node)

graph.add_node("efra", efra_node)

#Continue existing flow

graph.add_node("generate", test_case_node)

graph.add_node("review", review_node)

#Updated flow

graph.set_entry_point("fsd")

graph.add_edge("fsd", "regression")

graph.add_edge("regression", "report")

graph.add_edge("report", "disclosure")

graph.add edge("disclosure", "efra")

graph.add_edge("efra", "generate")

graph.add_edge("generate", "review")

graph.set finish_point("review")


def run_test_case_generation(fsd_path, report_folder, test_case_excel_path, efra_t5 path, disclosure_path):

initial state ("fsd path": fsd path, "report folder": report folder, "regression path": test_case_excel_path, "efra_t5_path": efra_t5_path, "disclosure_path":

disclosure_path}

final state graph.compile().invoke(initial_state)

fsd_path/fsd_docs/EFRA Reporting Layer Functional Specification.docx'

tableau_reports_path/tableau_reports'

test_case_excel_path/sample_test_cases/Regression_test_cases_Solai.xlsx

efra_t5 path/TS_extract/EFRA_T5_Table.csv'

disclosure path/derivative_excel/derivative_screen.xlsx

run_test_case_generation(fsd_path, tableau_reports_path, test_case_excel_path, efra_t5 path, disclosure path)