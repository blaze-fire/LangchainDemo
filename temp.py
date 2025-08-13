import os
from langchain.document_loaders import UnstructuredWordDocumentLoader, pandas
from langchain_openai import AzureOpenAI
from langgraph.graph import StateGraph
from crewai import Crew, Agent, Task  # pseudocode imports

# 1. Setup AzureOpenAI via LangChain
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "<your-azure-endpoint>"
os.environ["AZURE_OPENAI_API_KEY"] = "<your-api-key>"

llm = AzureOpenAI(
    deployment_name="your-deployment-name"
)

# 2. Define agent functions
def load_documents(state):
    state["fsd_text"] = UnstructuredWordDocumentLoader("fsd.docx").load()
    state["excel_samples"] = pandas.PandasExcelLoader("sample_tests.xlsx").load()
    # For .twbx, you might use custom parsing or Tableau LangChain tools
    state["twbx_data"] = load_twbx("report.twbx")  
    return state

def analyze_documents(state):
    prompt = (
        "Here are the FSD, test examples, and tableau output. "
        "Identify key areas to generate manual test cases."
        f"\nFSD:\n{state['fsd_text']}\n"
        f"Excel Samples:\n{state['excel_samples']}\n"
        f"Tableau Data:\n{state['twbx_data']}\n"
    )
    state["analysis"] = llm.invoke(prompt)
    return state

def generate_test_cases(state):
    prompt = (
        "Based on this analysis, generate manual test cases in bullet format:\n"
        f"{state['analysis']}"
    )
    state["test_cases"] = llm.invoke(prompt)
    return state

# 3. Build the LangGraph
graph = StateGraph()
graph.add_node("load", load_documents)
graph.add_node("analyze", analyze_documents)
graph.add_node("generate", generate_test_cases)
graph.add_edge("load", "analyze")
graph.add_edge("analyze", "generate")
compiled = graph.compile()

# 4. Use CrewAI to define agents and run
loader_agent = Agent(role="loader", fn=load_documents)
analyzer_agent = Agent(role="analyzer", fn=analyze_documents)
writer_agent = Agent(role="writer", fn=generate_test_cases)

crew = Crew(agents=[loader_agent, analyzer_agent, writer_agent], graph=compiled)
result_state = crew.run(initial_state={})

print("Generated Manual Test Cases:\n", result_state["test_cases"])




import zipfile
import tempfile
import os
import xml.etree.ElementTree as ET

def load_twbx(twbx_path: str) -> str:
    """
    Extracts metadata from a Tableau .twbx file for analysis.
    Returns a string summary of dashboards, sheets, fields, and filters.
    """
    if not os.path.exists(twbx_path):
        raise FileNotFoundError(f"{twbx_path} not found.")

    # Step 1: Unzip to temp folder
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(twbx_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Step 2: Find .twb file
        twb_file = None
        for root, _, files in os.walk(tmpdir):
            for f in files:
                if f.lower().endswith(".twb"):
                    twb_file = os.path.join(root, f)
                    break
        if not twb_file:
            raise FileNotFoundError("No .twb file found in .twbx")

        # Step 3: Parse XML
        tree = ET.parse(twb_file)
        root = tree.getroot()

        # Step 4: Extract key metadata
        ns = {"t": "http://www.tableausoftware.com/xml/user"}  # Tableau XML namespace
        dashboards = []
        worksheets = []

        # Dashboards
        for dashboard in root.findall(".//t:dashboard", ns):
            dashboards.append(dashboard.get("name"))

        # Worksheets
        for worksheet in root.findall(".//t:worksheet", ns):
            worksheets.append(worksheet.get("name"))

        # Fields
        fields = []
        for column in root.findall(".//t:column", ns):
            name = column.get("name")
            datatype = column.get("datatype")
            if name:
                fields.append(f"{name} ({datatype})")

        # Filters
        filters = []
        for filter_elem in root.findall(".//t:filter", ns):
            fname = filter_elem.get("name") or filter_elem.get("field")
            if fname:
                filters.append(fname)

        # Step 5: Summarize
        summary = (
            f"Dashboards: {', '.join(dashboards) or 'None'}\n"
            f"Worksheets: {', '.join(worksheets) or 'None'}\n"
            f"Fields: {', '.join(fields) or 'None'}\n"
            f"Filters: {', '.join(filters) or 'None'}"
        )

        return summary

