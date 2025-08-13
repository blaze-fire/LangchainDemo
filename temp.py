"""
Test Case Generator using CrewAI, LangGraph, and Azure OpenAI
This script generates manual test cases for Tableau reports based on FSD documents and sample test files.
Updated to fix all deprecated libraries and functions as of 2024-2025.
"""

import os
import json
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from pathlib import Path

# Azure OpenAI imports (Updated)
from openai import AzureOpenAI

# LangChain imports (Updated for 0.3.x compatibility)
from langchain_community.document_loaders import (
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# CrewAI imports (Updated - no changes needed as CrewAI is stable)
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# LangGraph imports (Updated with StateGraph)
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

# Configuration
@dataclass
class Config:
    """Configuration for Azure OpenAI and file paths"""
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/")
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key")
    api_version: str = "2024-10-21"  # Updated to latest GA API version
    deployment_name: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")  # Updated to current model
    
    fsd_path: str = "path/to/fsd_document.docx"
    excel_path: str = "path/to/sample_tests.xlsx"
    tableau_path: str = "path/to/tableau_report.twbx"


class TableauWorkbookLoader:
    """Custom loader for Tableau .twbx files"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load(self) -> List[Document]:
        """Extract and parse Tableau workbook content"""
        documents = []
        
        try:
            # .twbx is a packaged workbook (zip file)
            with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
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
                        
                        # Create document
                        doc_content = json.dumps(content, indent=2)
                        documents.append(
                            Document(
                                page_content=doc_content,
                                metadata={"source": self.file_path, "worksheet": name}
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
                    metadata={"source": self.file_path, "error": str(e)}
                )
            )
        
        return documents


class DocumentAnalysisTool(BaseTool):
    """Tool for analyzing documents and extracting relevant information"""
    
    name: str = "document_analyzer"
    description: str = "Analyzes documents to extract test-relevant information"
    
    def __init__(self, documents: Dict[str, List[Document]]):
        super().__init__()
        self.documents = documents
    
    def _run(self, query: str) -> str:
        """Execute document analysis based on query"""
        results = []
        for doc_type, docs in self.documents.items():
            for doc in docs:
                if query.lower() in doc.page_content.lower():
                    results.append(f"[{doc_type}]: {doc.page_content[:500]}...")
        
        return "\n\n".join(results) if results else "No relevant information found."


class TestCaseGenerator:
    """Main class for generating test cases using CrewAI and LangGraph"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = self._initialize_azure_client()
        self.documents = self._load_documents()
        self.crew = self._setup_crew()
        self.workflow = self._setup_langgraph_workflow()
    
    def _initialize_azure_client(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client with updated configuration"""
        return AzureOpenAI(
            azure_endpoint=self.config.azure_endpoint,
            api_key=self.config.api_key,
            api_version=self.config.api_version
        )
    
    def _load_documents(self) -> Dict[str, List[Document]]:
        """Load all documents using appropriate loaders"""
        documents = {}
        
        # Load FSD document
        if os.path.exists(self.config.fsd_path):
            fsd_loader = Docx2txtLoader(self.config.fsd_path)
            documents["fsd"] = fsd_loader.load()
        
        # Load Excel sample tests
        if os.path.exists(self.config.excel_path):
            excel_loader = UnstructuredExcelLoader(self.config.excel_path)
            documents["excel"] = excel_loader.load()
        
        # Load Tableau workbook
        if os.path.exists(self.config.tableau_path):
            tableau_loader = TableauWorkbookLoader(self.config.tableau_path)
            documents["tableau"] = tableau_loader.load()
        
        # Split documents for better processing (Updated import path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
        for doc_type in documents:
            documents[doc_type] = text_splitter.split_documents(documents[doc_type])
        
        return documents
    
    def _create_llm_function(self):
        """Create a function that uses Azure OpenAI for LLM calls"""
        def llm_function(prompt: str) -> str:
            try:
                response = self.client.chat.completions.create(
                    model=self.config.deployment_name,
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
    
    def _setup_crew(self) -> Crew:
        """Setup CrewAI agents and tasks"""
        
        # Create document analysis tool
        doc_tool = DocumentAnalysisTool(self.documents)
        
        # Create LLM function for agents
        llm_function = self._create_llm_function()
        
        # Agent 1: Requirements Analyst
        requirements_analyst = Agent(
            role="Requirements Analyst",
            goal="Extract and understand functional requirements from FSD document",
            backstory="Expert in analyzing functional specification documents and identifying testable requirements",
            tools=[doc_tool],
            llm=llm_function,
            verbose=True
        )
        
        # Agent 2: Tableau Specialist
        tableau_specialist = Agent(
            role="Tableau Report Specialist",
            goal="Analyze Tableau workbook structure and identify test points",
            backstory="Specialist in Tableau reports, understanding worksheets, calculations, and data sources",
            tools=[doc_tool],
            llm=llm_function,
            verbose=True
        )
        
        # Agent 3: Test Case Designer
        test_designer = Agent(
            role="Senior Test Case Designer",
            goal="Create comprehensive, high-quality test cases",
            backstory="20+ years of experience in designing test cases that exceed human quality standards",
            tools=[doc_tool],
            llm=llm_function,
            verbose=True
        )
        
        # Task 1: Analyze Requirements
        analyze_requirements = Task(
            description="""
            Analyze the FSD document to:
            1. Extract all functional requirements
            2. Identify business rules and validation criteria
            3. Map requirements to testable scenarios
            4. List all critical user workflows
            """,
            agent=requirements_analyst,
            expected_output="Comprehensive list of requirements and test scenarios"
        )
        
        # Task 2: Analyze Tableau Report
        analyze_tableau = Task(
            description="""
            Analyze the Tableau workbook to:
            1. Identify all worksheets and dashboards
            2. Extract calculations and their logic
            3. Identify filters and parameters
            4. Map data sources and their relationships
            5. Identify visual elements requiring validation
            """,
            agent=tableau_specialist,
            expected_output="Detailed Tableau report analysis with test points"
        )
        
        # Task 3: Generate Test Cases
        generate_tests = Task(
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
            agent=test_designer,
            expected_output="Complete set of high-quality test cases",
            context=[analyze_requirements, analyze_tableau]
        )
        
        # Create and return crew
        return Crew(
            agents=[requirements_analyst, tableau_specialist, test_designer],
            tasks=[analyze_requirements, analyze_tableau, generate_tests],
            process=Process.sequential,
            verbose=True
        )
    
    def _setup_langgraph_workflow(self) -> StateGraph:
        """Setup LangGraph workflow using StateGraph for orchestrating the test generation process"""
        
        # Create StateGraph with WorkflowState
        workflow = StateGraph(WorkflowState)
        
        # Define state functions
        def load_state(state: WorkflowState) -> WorkflowState:
            """Initial state - load documents"""
            state["documents_loaded"] = True
            state["documents"] = self.documents
            print("✓ Documents loaded successfully")
            return state
        
        def analyze_state(state: WorkflowState) -> WorkflowState:
            """Analyze documents state"""
            print("🔍 Starting document analysis...")
            state["analysis_complete"] = True
            # Run crew analysis tasks using kickoff() method (still current in CrewAI)
            result = self.crew.kickoff()
            state["analysis_result"] = str(result)
            print("✓ Document analysis completed")
            return state
        
        def generate_state(state: WorkflowState) -> WorkflowState:
            """Generate test cases state"""
            print("⚙️ Generating test cases...")
            state["generation_complete"] = True
            # Test cases are generated as part of crew execution
            state["test_cases"] = state.get("analysis_result", "")
            print("✓ Test case generation completed")
            return state
        
        def validate_state(state: WorkflowState) -> WorkflowState:
            """Validate generated test cases"""
            print("✅ Validating test cases...")
            llm_function = self._create_llm_function()
            
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
    
    def generate_test_cases(self) -> str:
        """Main method to generate test cases using invoke() instead of deprecated methods"""
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
        
        # Use invoke() method which is current standard in LangGraph
        final_state = self.workflow.invoke(initial_state)
        
        # Extract test cases
        test_cases = final_state.get("validated_test_cases", "No test cases generated")
        
        return test_cases
    
    def format_test_cases(self, test_cases: str) -> List[Dict]:
        """Format test cases into structured format"""
        llm_function = self._create_llm_function()
        
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


def main():
    """Main execution function"""
    
    # Initialize configuration with updated defaults
    config = Config(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "your-api-key"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),  # Updated default model
        fsd_path="path/to/your/fsd_document.docx",
        excel_path="path/to/your/sample_tests.xlsx",
        tableau_path="path/to/your/tableau_report.twbx"
    )
    
    # Create generator instance
    generator = TestCaseGenerator(config)
    
    # Generate test cases
    print("\n" + "="*80)
    print("GENERATING TEST CASES FOR TABLEAU REPORT")
    print("="*80 + "\n")
    
    test_cases = generator.generate_test_cases()
    
    # Format test cases
    formatted_cases = generator.format_test_cases(test_cases)
    
    # Print results
    print("\n" + "="*80)
    print("GENERATED TEST CASES")
    print("="*80 + "\n")
    
    if isinstance(formatted_cases, list) and len(formatted_cases) > 0:
        for i, test_case in enumerate(formatted_cases, 1):
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
                print(f"\nTest Case {i}:")
                print(json.dumps(test_case, indent=2))
    else:
        print(test_cases)
    
    print("\n" + "="*80)
    print("TEST CASE GENERATION COMPLETE")
    print("="*80)


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
    
    Note: Ensure you're using Python >=3.10 <3.14 for CrewAI compatibility
    """
    
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease ensure:")
        print("1. All required packages are installed with correct versions")
        print("2. Azure OpenAI credentials are set correctly")
        print("3. API version is supported (using 2024-10-21)")
        print("4. Model deployment exists and is accessible")
        print("5. File paths are correct and files exist")
        print("6. Python version is >=3.10 <3.14")
