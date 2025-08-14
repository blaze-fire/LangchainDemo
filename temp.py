import pandas as pd
from crewai import Task, Crew
import json

def review_node(state):
    # Get all context for comprehensive review
    fsd_rules = state.get('fsd_rules', '')
    report_kpis = state.get('report_kpis', '')
    regression_insights = state.get('regression_insights', '')
    test_cases = state.get('test_cases', '')
    
    # Read the sample regression Excel file to understand structure
    sample_regression_path = state["regression_path"]  # Your sample file
    sample_df = pd.read_excel(sample_regression_path)
    
    # Extract column structure and format information
    column_names = list(sample_df.columns)
    sample_rows = sample_df.head(3).to_dict('records')  # Get first 3 rows as examples
    
    # Convert sample structure to string for the agent
    column_structure = {
        "columns": column_names,
        "sample_format": sample_rows,
        "total_columns": len(column_names),
        "data_types": {col: str(sample_df[col].dtype) for col in column_names}
    }
    
    review_task = Task(
        description=f"""
        Review and finalize the generated test cases, then format them to match the exact structure of the sample regression Excel file.
        
        CRITICAL REQUIREMENT: The output must be formatted as an Excel-ready structure with the EXACT same columns as the sample file.
        
        Original Business Rules (FSD):
        {fsd_rules}
        
        Report KPIs and Logic:
        {report_kpis}
        
        Regression Insights:
        {regression_insights}
        
        Generated Test Cases:
        {test_cases}
        
        SAMPLE EXCEL STRUCTURE TO MATCH:
        Column Names: {column_names}
        Sample Format: {json.dumps(sample_rows, indent=2)}
        Total Columns Required: {len(column_names)}
        
        INSTRUCTIONS:
        1. Review the test cases for completeness and quality
        2. Format each test case to fit the exact column structure shown above
        3. Ensure every test case has a value for each column (use "N/A" or appropriate defaults if needed)
        4. Match the data format and style of the sample rows
        5. Output should be a JSON array where each object represents one test case row
        6. Each object must have keys that exactly match the column names: {column_names}
        
        OUTPUT FORMAT REQUIRED:
        Return a JSON array like this:
        [
            {{"{column_names[0]}": "value1", "{column_names[1]}": "value2", ...}},
            {{"{column_names[0]}": "value1", "{column_names[1]}": "value2", ...}},
            ...
        ]
        
        Review criteria:
        1. Completeness - Do test cases cover all business rules from FSD?
        2. Coverage - Are all KPIs and report logic tested?
        3. Format Compliance - Does each test case match the required column structure?
        4. Data Quality - Are all fields properly populated?
        5. Consistency - Is the format consistent across all test cases?
        """,
        agent=qa_reviewer,
        expected_output="JSON array of test cases formatted to match the exact Excel column structure"
    )
    
    crew = Crew(
        agents=[qa_reviewer],
        tasks=[review_task],
        process="sequential"
    )
    
    result = crew.kickoff()
    
    try:
        # Parse the JSON response from the agent
        if isinstance(result, str):
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                test_cases_data = json.loads(json_str)
            else:
                raise ValueError("No valid JSON array found in response")
        else:
            test_cases_data = result
        
        # Create DataFrame with exact same structure as sample
        output_df = pd.DataFrame(test_cases_data)
        
        # Ensure all required columns are present
        for col in column_names:
            if col not in output_df.columns:
                output_df[col] = "N/A"
        
        # Reorder columns to match sample file
        output_df = output_df[column_names]
        
        # Save to Excel file
        output_path = state.get("output_path", "generated_test_cases.xlsx")
        output_df.to_excel(output_path, index=False)
        
        return {
            **state,
            "final_test_cases": result,
            "test_cases_excel": output_path,
            "test_cases_dataframe": output_df.to_dict('records')
        }
        
    except Exception as e:
        print(f"Error formatting Excel output: {e}")
        # Fallback: return original result
        return {
            **state,
            "final_test_cases": result,
            "formatting_error": str(e)
        }

# Alternative helper function to ensure better formatting
def format_test_cases_to_excel(test_cases_json, sample_excel_path, output_path):
    """
    Helper function to convert test cases JSON to Excel with same format as sample
    """
    # Read sample structure
    sample_df = pd.read_excel(sample_excel_path)
    column_names = list(sample_df.columns)
    
    # Create output DataFrame
    output_df = pd.DataFrame(test_cases_json)
    
    # Ensure all columns exist
    for col in column_names:
        if col not in output_df.columns:
            output_df[col] = ""
    
    # Reorder and select only required columns
    output_df = output_df[column_names]
    
    # Apply any data type formatting from sample
    for col in column_names:
        if col in sample_df.columns:
            if sample_df[col].dtype == 'int64':
                output_df[col] = pd.to_numeric(output_df[col], errors='coerce').fillna(0).astype(int)
            elif sample_df[col].dtype == 'float64':
                output_df[col] = pd.to_numeric(output_df[col], errors='coerce').fillna(0.0)
    
    # Save to Excel
    output_df.to_excel(output_path, index=False)
    return output_df

# Updated state schema should include output path
# Add this to your initial state:
# {
#     "regression_path": "sample_regression.xlsx",
#     "output_path": "generated_test_cases.xlsx",
#     ...
# }
