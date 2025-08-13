"""
Test Case Generator using CrewAI, LangGraph, and Azure OpenAI
Functional programming approach - no classes, only functions
Updated to fix all deprecated libraries and functions as of 2024-2025.
"""

import os
import json
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, TypedDict, Callable
from pathlib import Path

# Azure OpenAI imports
from openai import AzureOpenAI

# LangChain imports
from langchain_community.document_loaders import (
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Additional imports for Excel handling
try:
    import pandas as pd
    import openpyxl
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# CrewAI imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool, tool

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

# Define State Type for LangGraph
class WorkflowState(TypedDict):
    documents_loaded: bool
    documents: Dict[str, List[Document]]
    analysis_complete: bool
    analysis_result: str
    generation_complete: bool
    test_cases: str
    validation_complete: bool
    validated_test_cases: str

# Configuration type
ConfigDict = TypedDict('ConfigDict', {
    'azure_endpoint': str,
    'api_key': str,
    'api_version': str,
    'deployment_name': str,
    'fsd_path': str,
    'excel_path': str,
    'tableau_path': str
})

def create_config() -> ConfigDict:
    """Create configuration dictionary for Azure OpenAI and file paths"""
    return {
        'azure_endpoint': os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/"),
        'api_key': os.getenv("AZURE_OPENAI_API_KEY", "your-api-key"),
        'api_version': "2024-10-21",
        'deployment_name': os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        'fsd_path': "path/to/fsd_document.docx",
        'excel_path': "path/to/sample_tests.xlsx",
        'tableau_path': "path/to/tableau_report.twbx"
    }

def initialize_azure_client(config: ConfigDict) -> AzureOpenAI:
    """Initialize Azure OpenAI client with configuration"""
    return AzureOpenAI(
        azure_endpoint=config['azure_endpoint'],
        api_key=config['api_key'],
        api_version=config['api_version']
    )

def load_tableau_workbook(file_path: str) -> List[Document]:
    """Extract and parse Tableau workbook content"""
    documents = []
    
    try:
        # .twbx is a packaged workbook (zip file)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Extract to temporary directory
            temp_dir = Path("temp_tableau_extract")
            temp_dir.mkdir(exist_ok=True)
            zip_ref.extractall(temp_dir)
            
            # Find and parse the .twb file (XML)
            for file in temp_dir.glob("*.twb"):
                tree = ET.parse(file)
                root = tree.getroot()
                
                # Extract worksheets
                worksheets = root.findall(".//worksheet")
                for worksheet in worksheets:
                    name = worksheet.get('name', 'Unknown')
                    
                    # Extract relevant information
                    content = extract_worksheet_content(root, worksheet)
                    
                    # Create document
                    doc_content = json.dumps(content, indent=2)
                    documents.append(
                        Document(
                            page_content=doc_content,
                            metadata={"source": file_path, "worksheet": name}
                        )
                    )
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"Error loading Tableau workbook: {e}")
        # Return a basic document if parsing fails
        documents.append(
            Document(
                page_content=f"Failed to parse Tableau workbook: {str(e)}",
                metadata={"source": file_path, "error": str(e)}
            )
        )
    
    return documents

def extract_worksheet_content(root, worksheet) -> Dict[str, Any]:
    """Extract content from a worksheet element"""
    name = worksheet.get('name', 'Unknown')
    
    content = {
        "worksheet_name": name,
        "datasources": [],
        "calculations": [],
        "filters": [],
        "parameters": []
    }
    
    # Extract datasources
    datasources = root.findall(".//datasource")
    for ds in datasources:
        ds_name = ds.get('name', 'Unknown')
        content["datasources"].append(ds_name)
    
    # Extract calculated fields
    calculations = worksheet.findall(".//calculation")
    for calc in calculations:
        calc_name = calc.get('name', '')
        formula = calc.get('formula', '')
        if calc_name:
            content["calculations"].append({
                "name": calc_name,
                "formula": formula
            })
    
    # Extract filters
    filters = worksheet.findall(".//filter")
    for filter_elem in filters:
        filter_class = filter_elem.get('class', '')
        content["filters"].append(filter_class)
    
    # Extract parameters
    parameters = root.findall(".//parameter")
    for param in parameters:
        param_name = param.get('name', '')
        content["parameters"].append(param_name)
    
    return content

def load_excel_with_pandas(file_path: str) -> List[Document]:
    """Load Excel file using pandas as fallback"""
    if not PANDAS_AVAILABLE:
        print(f"Warning: Cannot load Excel file {file_path} - pandas not available")
        return []
    
    try:
        # Read all sheets from Excel file
        excel_file = pd.ExcelFile(file_path)
        documents = []
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Convert DataFrame to text
            content = f"Sheet: {sheet_name}\n\n"
            content += df.to_string(index=False)
            
            document = Document(
                page_content=content,
                metadata={"source": file_path, "sheet_name": sheet_name}
            )
            documents.append(document)
        
        return documents
    
    except Exception as e:
        print(f"Error loading Excel file with pandas: {e}")
        return []

def load_documents(config: ConfigDict) -> Dict[str, List[Document]]:
    """Load all documents using appropriate loaders with fallbacks"""
    documents = {}
    
    # Load FSD document
    if os.path.exists(config['fsd_path']):
        try:
            fsd_loader = Docx2txtLoader(config['fsd_path'])
            documents["fsd"] = fsd_loader.load()
        except Exception as e:
            print(f"Error loading FSD document: {e}")
            documents["fsd"] = []
    
    # Load Excel sample tests with fallback
    if os.path.exists(config['excel_path']):
        try:
            excel_loader = UnstructuredExcelLoader(config['excel_path'])
            documents["excel"] = excel_loader.load()
        except Exception as e:
            print(f"Error loading Excel with UnstructuredExcelLoader: {e}")
            print("Trying pandas fallback...")
            documents["excel"] = load_excel_with_pandas(config['excel_path'])
    
    # Load Tableau workbook
    if os.path.exists(config['tableau_path']):
        try:
            documents["tableau"] = load_tableau_workbook(config['tableau_path'])
        except Exception as e:
            print(f"Error loading Tableau workbook: {e}")
            documents["tableau"] = []
    
    # Split documents for better processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    
    for doc_type in documents:
        if documents[doc_type]:  # Only split if documents exist
            documents[doc_type] = text_splitter.split_documents(documents[doc_type])
    
    return documents

def create_llm_function(client: AzureOpenAI, deployment_name: str) -> Callable[[str], str]:
    """Create a function that uses Azure OpenAI for LLM calls"""
    def llm_function(prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert QA engineer specialized in creating comprehensive test cases."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    return llm_function

def create_document_analysis_tool(documents: Dict[str, List[Document]]):
    """Create a document analysis tool using CrewAI's @tool decorator"""
    
    @tool("document_analyzer")
    def analyze_documents(query: str) -> str:
        """Analyzes documents to extract test-relevant information based on query"""
        results = []
        for doc_type, docs in documents.items():
            for doc in docs:
                if query.lower() in doc.page_content.lower():
                    results.append(f"[{doc_type}]: {doc.page_content[:500]}...")
        
        return "\n\n".join(results) if results else "No relevant information found."
    
    return analyze_documents

def create_crewai_llm(config: ConfigDict) -> LLM:
    """Create CrewAI LLM configuration for Azure OpenAI"""
    return LLM(
        model=f"azure/{config['deployment_name']}",
        base_url=config['azure_endpoint'],
        api_key=config['api_key'],
        api_version=config['api_version']
    )

def create_requirements_analyst(doc_tool, llm: LLM) -> Agent:
    """Create requirements analyst agent"""
    return Agent(
        role="Requirements Analyst",
        goal="Extract and understand functional requirements from FSD document",
        backstory="Expert in analyzing functional specification documents and identifying testable requirements",
        tools=[doc_tool],
        llm=llm,
        verbose=True
    )

def create_tableau_specialist(doc_tool, llm: LLM) -> Agent:
    """Create Tableau specialist agent"""
    return Agent(
        role="Tableau Report Specialist",
        goal="Analyze Tableau workbook structure and identify test points",
        backstory="Specialist in Tableau reports, understanding worksheets, calculations, and data sources",
        tools=[doc_tool],
        llm=llm,
        verbose=True
    )

def create_test_designer(doc_tool, llm: LLM) -> Agent:
    """Create test case designer agent"""
    return Agent(
        role="Senior Test Case Designer",
        goal="Create comprehensive, high-quality test cases",
        backstory="20+ years of experience in designing test cases that exceed human quality standards",
        tools=[doc_tool],
        llm=llm,
        verbose=True
    )

def create_analyze_requirements_task(agent: Agent) -> Task:
    """Create requirements analysis task"""
    return Task(
        description="""
        Analyze the FSD document to:
        1. Extract all functional requirements
        2. Identify business rules and validation criteria
        3. Map requirements to testable scenarios
        4. List all critical user workflows
        """,
        agent=agent,
        expected_output="Comprehensive list of requirements and test scenarios"
    )

def create_analyze_tableau_task(agent: Agent) -> Task:
    """Create Tableau analysis task"""
    return Task(
        description="""
        Analyze the Tableau workbook to:
        1. Identify all worksheets and dashboards
        2. Extract calculations and their logic
        3. Identify filters and parameters
        4. Map data sources and their relationships
        5. Identify visual elements requiring validation
        """,
        agent=agent,
        expected_output="Detailed Tableau report analysis with test points"
    )

def create_generate_tests_task(agent: Agent, requirements_task: Task, tableau_task: Task) -> Task:
    """Create test case generation task"""
    return Task(
        description="""
        Generate comprehensive test cases that:
        1. Cover all functional requirements from FSD
        2. Validate all Tableau report components
        3. Include positive, negative, and edge cases
        4. Follow the format from sample Excel tests
        5. Include data validation tests
        6. Cover performance and usability aspects
        7. Ensure 100% requirement coverage
        
        Each test case must include:
        - Test Case ID
        - Test Scenario
        - Test Description
        - Prerequisites
        - Test Steps (detailed)
        - Expected Results
        - Test Data
        - Priority (Critical/High/Medium/Low)
        - Test Type (Functional/UI/Data/Performance)
        """,
        agent=agent,
        expected_output="Complete set of high-quality test cases",
        context=[requirements_task, tableau_task]
    )

def setup_crew(documents: Dict[str, List[Document]], config: ConfigDict) -> Crew:
    """Setup CrewAI agents and tasks using functional approach"""
    
    # Create CrewAI LLM configuration
    crewai_llm = create_crewai_llm(config)
    
    # Create document analysis tool (returns a decorated function)
    doc_tool = create_document_analysis_tool(documents)
    
    # Create agents (crewai_llm is passed to each agent)
    requirements_analyst = create_requirements_analyst(doc_tool, crewai_llm)
    tableau_specialist = create_tableau_specialist(doc_tool, crewai_llm)
    test_designer = create_test_designer(doc_tool, crewai_llm)
    
    # Create tasks
    analyze_requirements = create_analyze_requirements_task(requirements_analyst)
    analyze_tableau = create_analyze_tableau_task(tableau_specialist)
    generate_tests = create_generate_tests_task(test_designer, analyze_requirements, analyze_tableau)
    
    # Create and return crew
    return Crew(
        agents=[requirements_analyst, tableau_specialist, test_designer],
        tasks=[analyze_requirements, analyze_tableau, generate_tests],
        process=Process.sequential,
        verbose=True
    )

def create_load_state_function(documents: Dict[str, List[Document]]) -> Callable[[WorkflowState], WorkflowState]:
    """Create load state function"""
    def load_state(state: WorkflowState) -> WorkflowState:
        """Initial state - load documents"""
        state["documents_loaded"] = True
        state["documents"] = documents
        print("✓ Documents loaded successfully")
        return state
    
    return load_state

def create_analyze_state_function(crew: Crew) -> Callable[[WorkflowState], WorkflowState]:
    """Create analyze state function"""
    def analyze_state(state: WorkflowState) -> WorkflowState:
        """Analyze documents state"""
        print("🔍 Starting document analysis...")
        state["analysis_complete"] = True
        # Run crew analysis tasks
        result = crew.kickoff()
        state["analysis_result"] = str(result)
        print("✓ Document analysis completed")
        return state
    
    return analyze_state

def create_generate_state_function() -> Callable[[WorkflowState], WorkflowState]:
    """Create generate state function"""
    def generate_state(state: WorkflowState) -> WorkflowState:
        """Generate test cases state"""
        print("⚙️ Generating test cases...")
        state["generation_complete"] = True
        # Test cases are generated as part of crew execution
        state["test_cases"] = state.get("analysis_result", "")
        print("✓ Test case generation completed")
        return state
    
    return generate_state

def create_validate_state_function(llm_function: Callable[[str], str]) -> Callable[[WorkflowState], WorkflowState]:
    """Create validate state function"""
    def validate_state(state: WorkflowState) -> WorkflowState:
        """Validate generated test cases"""
        print("✅ Validating test cases...")
        
        validation_prompt = f"""
        Review the following test cases and ensure they meet these criteria:
        1. Complete coverage of all requirements
        2. Clear and unambiguous steps
        3. Measurable expected results
        4. Appropriate test data
        5. Logical flow and sequence
        
        Test Cases:
        {state.get('test_cases', '')}
        
        Provide a validation report and enhanced test cases if needed.
        """
        
        validation_result = llm_function(validation_prompt)
        state["validated_test_cases"] = validation_result
        state["validation_complete"] = True
        print("✓ Test case validation completed")
        return state
    
    return validate_state

def setup_langgraph_workflow(
    documents: Dict[str, List[Document]], 
    crew: Crew, 
    llm_function: Callable[[str], str]
) -> StateGraph:
    """Setup LangGraph workflow using functional approach"""
    
    # Create StateGraph with WorkflowState
    workflow = StateGraph(WorkflowState)
    
    # Create state functions
    load_state = create_load_state_function(documents)
    analyze_state = create_analyze_state_function(crew)
    generate_state = create_generate_state_function()
    validate_state = create_validate_state_function(llm_function)
    
    # Add nodes to StateGraph
    workflow.add_node("load", load_state)
    workflow.add_node("analyze", analyze_state)
    workflow.add_node("generate", generate_state)
    workflow.add_node("validate", validate_state)
    
    # Add edges
    workflow.add_edge("load", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", END)
    
    # Set entry point
    workflow.set_entry_point("load")
    
    return workflow.compile()

def generate_test_cases(workflow: StateGraph) -> str:
    """Generate test cases using the workflow"""
    print("Starting test case generation process...")
    
    # Execute workflow with proper initial state
    initial_state: WorkflowState = {
        "documents_loaded": False,
        "documents": {},
        "analysis_complete": False,
        "analysis_result": "",
        "generation_complete": False,
        "test_cases": "",
        "validation_complete": False,
        "validated_test_cases": ""
    }
    
    # Use invoke() method
    final_state = workflow.invoke(initial_state)
    
    # Extract test cases
    test_cases = final_state.get("validated_test_cases", "No test cases generated")
    
    return test_cases

def format_test_cases(test_cases: str, llm_function: Callable[[str], str]) -> List[Dict]:
    """Format test cases into structured format"""
    
    format_prompt = f"""
    Format the following test cases into a structured JSON format with these fields:
    - test_case_id
    - scenario
    - description
    - prerequisites
    - steps (array)
    - expected_results
    - test_data
    - priority
    - test_type
    
    Test Cases:
    {test_cases}
    
    Return only valid JSON array.
    """
    
    formatted_result = llm_function(format_prompt)
    
    try:
        return json.loads(formatted_result)
    except json.JSONDecodeError:
        # Return structured fallback if JSON parsing fails
        return [{"raw_test_cases": test_cases, "error": "Failed to parse JSON"}]

def print_test_case(test_case: Dict, index: int) -> None:
    """Print a single test case"""
    if "test_case_id" in test_case:
        print(f"\n--- Test Case {test_case['test_case_id']} ---")
        print(f"Scenario: {test_case.get('scenario', 'N/A')}")
        print(f"Description: {test_case.get('description', 'N/A')}")
        print(f"Priority: {test_case.get('priority', 'N/A')}")
        print(f"Type: {test_case.get('test_type', 'N/A')}")
        print(f"\nPrerequisites:")
        print(test_case.get('prerequisites', 'N/A'))
        print(f"\nTest Steps:")
        steps = test_case.get('steps', [])
        for j, step in enumerate(steps, 1):
            print(f"  {j}. {step}")
        print(f"\nExpected Results:")
        print(test_case.get('expected_results', 'N/A'))
        print(f"\nTest Data:")
        print(test_case.get('test_data', 'N/A'))
        print("-" * 40)
    else:
        print(f"\nTest Case {index}:")
        print(json.dumps(test_case, indent=2))

def print_test_results(formatted_cases: List[Dict], raw_test_cases: str) -> None:
    """Print formatted test results"""
    print("\n" + "="*80)
    print("GENERATED TEST CASES")
    print("="*80 + "\n")
    
    if isinstance(formatted_cases, list) and len(formatted_cases) > 0:
        for i, test_case in enumerate(formatted_cases, 1):
            print_test_case(test_case, i)
    else:
        print(raw_test_cases)
    
    print("\n" + "="*80)
    print("TEST CASE GENERATION COMPLETE")
    print("="*80)

def main():
    """Main execution function using only functional programming"""
    
    # Initialize configuration
    config = create_config()
    
    # Initialize Azure client (for validation tasks)
    client = initialize_azure_client(config)
    
    # Load documents
    documents = load_documents(config)
    
    # Create LLM function (for validation)
    llm_function = create_llm_function(client, config['deployment_name'])
    
    # Setup crew (uses CrewAI LLM configuration)
    crew = setup_crew(documents, config)
    
    # Setup workflow
    workflow = setup_langgraph_workflow(documents, crew, llm_function)
    
    # Generate test cases
    print("\n" + "="*80)
    print("GENERATING TEST CASES FOR TABLEAU REPORT")
    print("="*80 + "\n")
    
    test_cases = generate_test_cases(workflow)
    
    # Format test cases
    formatted_cases = format_test_cases(test_cases, llm_function)
    
    # Print results
    print_test_results(formatted_cases, test_cases)

if __name__ == "__main__":
    # Updated package installation instructions
    """
    Required packages (install with):
    pip install crewai==0.157.0
    pip install langgraph>=0.2.20
    pip install langchain>=0.3.0
    pip install langchain-community>=0.3.0
    pip install langchain-text-splitters>=0.3.0
    pip install openai>=1.0.0
    pip install docx2txt openpyxl unstructured python-docx
    pip install pandas  # For Excel file handling fallback
    pip install msoffcrypto  # For encrypted Office documents
    
    Note: Ensure you're using Python >=3.10 <3.14 for CrewAI compatibility
    
    If you get msoffcrypto error, install it with:
    pip install msoffcrypto-tool
    
    Alternative Excel libraries if issues persist:
    pip install xlrd xlsxwriter
    """
    
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease ensure:")
        print("1. All required packages are installed with correct versions")
        print("2. Install msoffcrypto: pip install msoffcrypto-tool")
        print("3. Azure OpenAI credentials are set correctly")
        print("4. API version is supported (using 2024-10-21)")
        print("5. Model deployment exists and is accessible")
        print("6. File paths are correct and files exist")
        print("7. Python version is >=3.10 <3.14")
        print("8. If Excel files are password-protected, ensure msoffcrypto is installed")
