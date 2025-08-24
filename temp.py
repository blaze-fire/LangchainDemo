# Tableau Test Case Generation System using LangGraph, LangChain, and OpenAI
# Using functional programming approach

import os
import json
from typing import Dict, List, Any, TypedDict, Annotated, Sequence
from enum import Enum
import operator

# Core imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint import MemorySaver

# Document processing
import pandas as pd
from docx import Document as DocxDocument
import openpyxl

# ============================================
# State Definition
# ============================================

class AgentState(TypedDict):
    """State that flows through the graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    fsd_content: str
    t5_data: Dict[str, Any]
    disclosure_mapping: Dict[str, Any]
    regression_samples: List[str]
    extracted_requirements: Dict[str, Any]
    generated_test_cases: List[Dict[str, Any]]
    reviewed_test_cases: List[Dict[str, Any]]
    current_agent: str
    iteration: int
    feedback: str

# ============================================
# Document Processing Functions
# ============================================

def process_fsd_document(file_path: str) -> str:
    """Extract and process FSD Word document"""
    try:
        doc = DocxDocument(file_path)
        content = []
        
        # Extract all paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text.strip())
        
        # Extract tables if any
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                if any(row_data):
                    table_data.append(" | ".join(row_data))
            if table_data:
                content.append("\nTable Data:\n" + "\n".join(table_data))
        
        return "\n\n".join(content)
    except Exception as e:
        return f"Error processing FSD document: {str(e)}"

def process_t5_extract(file_path: str) -> Dict[str, Any]:
    """Process T5 extract Excel file"""
    try:
        df = pd.read_excel(file_path, sheet_name=None)
        t5_data = {}
        
        for sheet_name, sheet_df in df.items():
            # Convert DataFrame to dictionary for easier processing
            sheet_data = {
                "columns": sheet_df.columns.tolist(),
                "data": sheet_df.to_dict('records'),
                "summary": {
                    "row_count": len(sheet_df),
                    "column_count": len(sheet_df.columns),
                    "data_types": sheet_df.dtypes.to_dict()
                }
            }
            t5_data[sheet_name] = sheet_data
        
        return t5_data
    except Exception as e:
        return {"error": f"Error processing T5 extract: {str(e)}"}

def process_disclosure_mapping(file_path: str) -> Dict[str, Any]:
    """Process disclosure mapping Excel file"""
    try:
        df = pd.read_excel(file_path)
        
        mapping_data = {
            "mappings": df.to_dict('records'),
            "columns": df.columns.tolist(),
            "unique_disclosures": df['Disclosure'].unique().tolist() if 'Disclosure' in df.columns else [],
            "unique_reports": df['Report'].unique().tolist() if 'Report' in df.columns else []
        }
        
        return mapping_data
    except Exception as e:
        return {"error": f"Error processing disclosure mapping: {str(e)}"}

def process_regression_test_cases(file_path: str) -> List[str]:
    """Process regression test cases - keeping them minimal to reduce influence"""
    try:
        # Assuming regression test cases are in Excel format
        df = pd.read_excel(file_path)
        
        # Extract only the structure/format, not the actual content
        test_case_structure = []
        
        if 'Test_Case' in df.columns:
            # Get only 2-3 samples for format reference
            samples = df['Test_Case'].head(3).tolist()
            test_case_structure = [f"Format reference: {tc[:100]}..." for tc in samples]
        
        return test_case_structure
    except Exception as e:
        return [f"Error processing regression tests: {str(e)}"]

# ============================================
# Agent Functions
# ============================================

def create_fsd_analyzer_agent(llm: ChatOpenAI):
    """Agent to analyze FSD document and extract requirements"""
    
    def analyze_fsd(state: AgentState) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert FSD document analyzer for Tableau reports.
            Your task is to extract ALL critical information from the FSD document that should be tested.
            
            Focus on:
            1. Business rules and validations
            2. Data transformations and calculations
            3. Report layouts and visualizations
            4. Filters and parameters
            5. User interactions and workflows
            6. Performance requirements
            7. Data quality checks
            8. Security and access controls
            
            Extract comprehensive requirements that will form the basis of test cases.
            Be thorough and capture every testable requirement."""),
            ("human", "FSD Content:\n{fsd_content}\n\nExtract all testable requirements:")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"fsd_content": state["fsd_content"]})
        
        requirements = {
            "business_rules": [],
            "data_validations": [],
            "ui_requirements": [],
            "calculations": [],
            "filters": [],
            "performance": [],
            "security": []
        }
        
        # Parse response and categorize requirements
        content = response.content if hasattr(response, 'content') else str(response)
        lines = content.split('\n')
        current_category = "business_rules"
        
        for line in lines:
            line = line.strip()
            if "business rule" in line.lower():
                current_category = "business_rules"
            elif "validation" in line.lower():
                current_category = "data_validations"
            elif "ui" in line.lower() or "layout" in line.lower():
                current_category = "ui_requirements"
            elif "calculation" in line.lower():
                current_category = "calculations"
            elif "filter" in line.lower():
                current_category = "filters"
            elif "performance" in line.lower():
                current_category = "performance"
            elif "security" in line.lower():
                current_category = "security"
            
            if line and not line.startswith('#'):
                requirements[current_category].append(line)
        
        state["extracted_requirements"] = requirements
        state["messages"].append(AIMessage(content=f"Extracted requirements from FSD: {json.dumps(requirements, indent=2)}"))
        return state
    
    return analyze_fsd

def create_t5_analyzer_agent(llm: ChatOpenAI):
    """Agent to analyze T5 extract data"""
    
    def analyze_t5(state: AgentState) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analyst specializing in T5 extracts for Tableau testing.
            Analyze the T5 data structure and identify:
            1. Key data fields and their relationships
            2. Data types and formats
            3. Potential data quality issues to test
            4. Edge cases based on data patterns
            5. Aggregation and grouping scenarios
            6. Null/empty value handling
            7. Data range validations
            
            Generate specific test scenarios based on the actual data structure."""),
            ("human", "T5 Data Structure:\n{t5_data}\n\nIdentify test scenarios based on this data:")
        ])
        
        chain = prompt | llm
        t5_summary = json.dumps(state["t5_data"], indent=2)[:3000]  # Limit size for LLM
        response = chain.invoke({"t5_data": t5_summary})
        
        # Update requirements with T5-specific test scenarios
        if "extracted_requirements" not in state:
            state["extracted_requirements"] = {}
        
        state["extracted_requirements"]["data_scenarios"] = response.content if hasattr(response, 'content') else str(response)
        state["messages"].append(AIMessage(content=f"T5 analysis completed"))
        return state
    
    return analyze_t5

def create_disclosure_mapping_agent(llm: ChatOpenAI):
    """Agent to analyze disclosure mapping"""
    
    def analyze_mapping(state: AgentState) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in Tableau disclosure mapping analysis.
            Based on the disclosure mapping data:
            1. Identify all report-disclosure relationships
            2. Define test cases for each mapping
            3. Verify data flow between reports and disclosures
            4. Check for consistency across mappings
            5. Validate business logic in mappings
            
            Focus on creating test cases that verify the accuracy of these mappings."""),
            ("human", "Disclosure Mapping:\n{mapping_data}\n\nGenerate mapping-specific test requirements:")
        ])
        
        chain = prompt | llm
        mapping_summary = json.dumps(state["disclosure_mapping"], indent=2)[:3000]
        response = chain.invoke({"mapping_data": mapping_summary})
        
        if "extracted_requirements" not in state:
            state["extracted_requirements"] = {}
        
        state["extracted_requirements"]["mapping_tests"] = response.content if hasattr(response, 'content') else str(response)
        state["messages"].append(AIMessage(content=f"Disclosure mapping analysis completed"))
        return state
    
    return analyze_mapping

def create_test_generator_agent(llm: ChatOpenAI):
    """Main agent to generate test cases"""
    
    def generate_tests(state: AgentState) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior QA engineer generating comprehensive test cases for Tableau reports.
            
            PRIORITY: Base test cases PRIMARILY on:
            1. FSD requirements (60% weight)
            2. T5 data structure and scenarios (25% weight)
            3. Disclosure mappings (10% weight)
            4. Regression test format only (5% weight - structure only, not content)
            
            Generate test cases with:
            - Test Case ID
            - Test Case Name
            - Description
            - Pre-conditions
            - Test Steps (detailed)
            - Expected Results
            - Test Data
            - Priority (High/Medium/Low)
            - Category
            
            Create diverse test cases covering:
            - Functional testing
            - Data validation
            - UI/UX testing
            - Performance testing
            - Security testing
            - Integration testing
            - Edge cases
            
            IMPORTANT: Focus on NEW requirements from FSD, not historical patterns."""),
            ("human", """Requirements: {requirements}
            
            Regression Format Reference (use format only): {regression_samples}
            
            Generate comprehensive test cases based primarily on the requirements:""")
        ])
        
        chain = prompt | llm
        
        # Prepare requirements summary
        req_summary = json.dumps(state["extracted_requirements"], indent=2)[:5000]
        
        # Minimal regression reference
        regression_ref = state["regression_samples"][:2] if state["regression_samples"] else ["No format reference"]
        
        response = chain.invoke({
            "requirements": req_summary,
            "regression_samples": "\n".join(regression_ref)
        })
        
        # Parse response into structured test cases
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Simple parsing - in production, use more sophisticated parsing
        test_cases = []
        current_test = {}
        
        for line in content.split('\n'):
            if "Test Case ID:" in line:
                if current_test:
                    test_cases.append(current_test)
                current_test = {"id": line.replace("Test Case ID:", "").strip()}
            elif "Test Case Name:" in line:
                current_test["name"] = line.replace("Test Case Name:", "").strip()
            elif "Description:" in line:
                current_test["description"] = line.replace("Description:", "").strip()
            elif "Priority:" in line:
                current_test["priority"] = line.replace("Priority:", "").strip()
            elif "Category:" in line:
                current_test["category"] = line.replace("Category:", "").strip()
        
        if current_test:
            test_cases.append(current_test)
        
        state["generated_test_cases"] = test_cases
        state["messages"].append(AIMessage(content=f"Generated {len(test_cases)} test cases"))
        return state
    
    return generate_tests

def create_review_agent(llm: ChatOpenAI):
    """Agent to review and enhance test cases"""
    
    def review_tests(state: AgentState) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a QA lead reviewing test cases for completeness and quality.
            
            Review criteria:
            1. Coverage of all FSD requirements
            2. Proper test case structure and clarity
            3. Testability and feasibility
            4. No redundancy or gaps
            5. Alignment with T5 data and disclosure mappings
            6. Clear expected results
            7. Appropriate priority levels
            
            Provide:
            - Overall quality score (1-10)
            - Coverage assessment
            - Improvements made
            - Recommendations for additional tests
            
            ENSURE: Test cases are NOT copies of regression tests but new cases based on current requirements."""),
            ("human", """Generated Test Cases: {test_cases}
            
            Original Requirements: {requirements}
            
            Review and enhance these test cases:""")
        ])
        
        chain = prompt | llm
        
        response = chain.invoke({
            "test_cases": json.dumps(state["generated_test_cases"], indent=2)[:4000],
            "requirements": json.dumps(state["extracted_requirements"], indent=2)[:2000]
        })
        
        # Add review feedback
        state["reviewed_test_cases"] = state["generated_test_cases"]
        state["feedback"] = response.content if hasattr(response, 'content') else str(response)
        state["messages"].append(AIMessage(content=f"Review completed: {state['feedback'][:200]}..."))
        return state
    
    return review_tests

# ============================================
# Graph Construction
# ============================================

def build_test_generation_graph(llm: ChatOpenAI):
    """Build the LangGraph workflow"""
    
    # Create agents
    fsd_analyzer = create_fsd_analyzer_agent(llm)
    t5_analyzer = create_t5_analyzer_agent(llm)
    mapping_analyzer = create_disclosure_mapping_agent(llm)
    test_generator = create_test_generator_agent(llm)
    reviewer = create_review_agent(llm)
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("fsd_analysis", fsd_analyzer)
    workflow.add_node("t5_analysis", t5_analyzer)
    workflow.add_node("mapping_analysis", mapping_analyzer)
    workflow.add_node("test_generation", test_generator)
    workflow.add_node("review", reviewer)
    
    # Define edges
    workflow.set_entry_point("fsd_analysis")
    workflow.add_edge("fsd_analysis", "t5_analysis")
    workflow.add_edge("t5_analysis", "mapping_analysis")
    workflow.add_edge("mapping_analysis", "test_generation")
    workflow.add_edge("test_generation", "review")
    workflow.add_edge("review", END)
    
    # Compile
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# ============================================
# Main Execution Function
# ============================================

def generate_tableau_test_cases(
    fsd_path: str,
    t5_path: str,
    mapping_path: str,
    regression_path: str,
    openai_api_key: str,
    model: str = "gpt-4-turbo-preview"
) -> Dict[str, Any]:
    """Main function to generate test cases"""
    
    # Initialize LLM
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model=model,
        temperature=0.3  # Lower temperature for more consistent output
    )
    
    # Process input documents
    print("Processing input documents...")
    fsd_content = process_fsd_document(fsd_path)
    t5_data = process_t5_extract(t5_path)
    disclosure_mapping = process_disclosure_mapping(mapping_path)
    regression_samples = process_regression_test_cases(regression_path)
    
    # Build graph
    print("Building workflow graph...")
    app = build_test_generation_graph(llm)
    
    # Prepare initial state
    initial_state = {
        "messages": [HumanMessage(content="Starting test case generation")],
        "fsd_content": fsd_content,
        "t5_data": t5_data,
        "disclosure_mapping": disclosure_mapping,
        "regression_samples": regression_samples,
        "extracted_requirements": {},
        "generated_test_cases": [],
        "reviewed_test_cases": [],
        "current_agent": "fsd_analysis",
        "iteration": 0,
        "feedback": ""
    }
    
    # Run the workflow
    print("Executing workflow...")
    config = {"configurable": {"thread_id": "test_generation_1"}}
    
    final_state = None
    for output in app.stream(initial_state, config):
        for key, value in output.items():
            print(f"Completed: {key}")
            final_state = value
    
    return {
        "test_cases": final_state.get("reviewed_test_cases", []),
        "requirements": final_state.get("extracted_requirements", {}),
        "feedback": final_state.get("feedback", ""),
        "messages": [msg.content for msg in final_state.get("messages", [])]
    }

# ============================================
# Export Functions
# ============================================

def export_test_cases_to_excel(test_cases: List[Dict], output_path: str):
    """Export test cases to Excel format"""
    df = pd.DataFrame(test_cases)
    df.to_excel(output_path, index=False)
    print(f"Test cases exported to {output_path}")

def export_test_cases_to_json(test_cases: List[Dict], output_path: str):
    """Export test cases to JSON format"""
    with open(output_path, 'w') as f:
        json.dump(test_cases, f, indent=2)
    print(f"Test cases exported to {output_path}")

# ============================================
# Usage Example
# ============================================

def main():
    """Example usage"""
    
    # Configuration
    config = {
        "fsd_path": "path/to/fsd_document.docx",
        "t5_path": "path/to/t5_extract.xlsx",
        "mapping_path": "path/to/disclosure_mapping.xlsx",
        "regression_path": "path/to/regression_tests.xlsx",
        "openai_api_key": "your-openai-api-key",
        "model": "gpt-4-turbo-preview"
    }
    
    # Generate test cases
    results = generate_tableau_test_cases(**config)
    
    # Export results
    export_test_cases_to_excel(
        results["test_cases"],
        "generated_test_cases.xlsx"
    )
    
    export_test_cases_to_json(
        results["test_cases"],
        "generated_test_cases.json"
    )
    
    # Print summary
    print("\n" + "="*50)
    print("Test Generation Summary")
    print("="*50)
    print(f"Total test cases generated: {len(results['test_cases'])}")
    print(f"\nReview Feedback:\n{results['feedback']}")
    
    return results

if __name__ == "__main__":
    results = main()
