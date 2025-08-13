# Updated Agents
report_analyst = Agent(
    name="Tableau Report Analyst",
    role="BI Dashboard Analyst",
    goal="Extract KPIs, calculated fields, filters, and data sources from Tableau reports",
    backstory="Skilled in reverse-engineering Tableau dashboards to understand business logic and metrics",
    llm=create_llm(),
    verbose=True,
    allow_delegation=False
)

regression_analyst = Agent(
    name="Regression Test Case Analyst",
    role="Regression Test Case Expert",
    goal="Analyze existing regression test cases and use them to guide generation of new test cases",
    backstory="Experienced in identifying reusable logic and patterns from historical test cases to improve future coverage",
    llm=create_llm(),
    verbose=True,
    allow_delegation=False
)

test_designer = Agent(
    name="Test Case Generator",
    role="QA Automation Specialist",
    goal="Design comprehensive test cases based on business rules and report logic, covering edge cases and validations",
    backstory="Experienced in translating business logic into robust test scenarios for data validation and reporting accuracy",
    llm=create_llm(),
    verbose=True,
    allow_delegation=False
)

qa_reviewer = Agent(
    name="QA Reviewer",
    role="Senior QA Lead",
    goal="Review test cases for completeness, clarity, and alignment with business requirements",
    backstory="Ensures that test cases meet quality standards and cover all functional and reporting aspects",
    llm=create_llm(),
    verbose=True,
    allow_delegation=False
)

# Updated Node Functions
def regression_node(state):
    regression_path = state["regression_path"]
    regression_text = extract_text_from_excel(regression_path)
    
    # Get FSD rules from previous node for context
    fsd_context = state.get("fsd_rules", "")
    
    regression_task = Task(
        description=f"""
        Analyze the regression test cases and extract reusable logic and patterns.
        
        Context from FSD Analysis:
        {fsd_context}
        
        Regression Test Cases:
        {regression_text}
        
        Extract:
        1. Common test patterns and logic
        2. Data validation rules being tested
        3. Edge cases covered in existing tests
        4. Reusable test components and utilities
        5. Gaps that could be addressed in new test cases
        
        Focus on identifying patterns that align with the business rules from the FSD.
        """,
        agent=regression_analyst,
        expected_output="Structured analysis of regression test patterns, reusable logic, and identified gaps"
    )
    
    crew = Crew(
        agents=[regression_analyst],
        tasks=[regression_task],
        process="sequential"
    )
    
    result = crew.kickoff()
    insights = result if isinstance(result, str) else str(result)
    
    return {
        **state,
        "regression_insights": insights
    }

def tableau_node(state):
    report_folder = state["report_folder"]
    combined_report_text = ""
    
    for file_name in os.listdir(report_folder):
        if file_name.endswith(".twbx"):
            file_path = os.path.join(report_folder, file_name)
            report_text = extract_logic_from_twbx(file_path)
            combined_report_text += f"\n--- Report: {file_name} ---\n{report_text}\n"
    
    # Get context from previous nodes
    fsd_context = state.get("fsd_rules", "")
    regression_context = state.get("regression_insights", "")
    
    tableau_task = Task(
        description=f"""
        Extract KPIs, calculated fields, filters, and business logic from Tableau reports.
        
        Context from FSD Analysis:
        {fsd_context}
        
        Context from Regression Analysis:
        {regression_context}
        
        Tableau Reports Content:
        {combined_report_text}
        
        Extract and analyze:
        1. Key Performance Indicators (KPIs) and metrics
        2. Calculated fields and their business logic
        3. Filters and parameter logic
        4. Data source relationships and joins
        5. Dashboard interactions and dependencies
        6. Validation rules embedded in reports
        
        Cross-reference with FSD business rules to ensure alignment and identify any discrepancies.
        """,
        agent=report_analyst,
        expected_output="Comprehensive analysis of Tableau report logic, KPIs, and business rules with cross-references to FSD"
    )
    
    crew = Crew(
        agents=[report_analyst],
        tasks=[tableau_task],
        process="sequential"
    )
    
    result = crew.kickoff()
    kpis = result if isinstance(result, str) else str(result)
    
    return {
        **state,
        "report_kpis": kpis
    }

def test_case_node(state):
    # Gather all context from previous nodes
    fsd_rules = state.get('fsd_rules', '')
    report_kpis = state.get('report_kpis', '')
    regression_insights = state.get('regression_insights', '')
    
    test_generation_task = Task(
        description=f"""
        Generate comprehensive test cases using all available context from the workflow.
        
        Business Rules from FSD:
        {fsd_rules}
        
        KPIs and Report Logic:
        {report_kpis}
        
        Regression Test Insights:
        {regression_insights}
        
        Generate test cases that:
        1. Validate all business rules identified in the FSD
        2. Test KPIs and calculated fields from Tableau reports
        3. Incorporate patterns and logic from regression analysis
        4. Cover positive, negative, and edge case scenarios
        5. Include data validation and boundary testing
        6. Test integration points and dependencies
        7. Validate report accuracy and data consistency
        
        Structure test cases with:
        - Test Case ID and Name
        - Objective/Purpose
        - Pre-conditions
        - Test Steps
        - Expected Results
        - Test Data Requirements
        - Priority Level
        """,
        agent=test_designer,
        expected_output="Comprehensive set of structured test cases covering functional, data validation, and reporting scenarios"
    )
    
    crew = Crew(
        agents=[test_designer],
        tasks=[test_generation_task],
        process="sequential"
    )
    
    result = crew.kickoff()
    test_cases = result if isinstance(result, str) else str(result)
    
    return {
        **state,
        "test_cases": test_cases
    }

def review_node(state):
    # Get all context for comprehensive review
    fsd_rules = state.get('fsd_rules', '')
    report_kpis = state.get('report_kpis', '')
    regression_insights = state.get('regression_insights', '')
    test_cases = state.get('test_cases', '')
    
    review_task = Task(
        description=f"""
        Review and finalize the generated test cases for completeness and quality.
        
        Original Business Rules (FSD):
        {fsd_rules}
        
        Report KPIs and Logic:
        {report_kpis}
        
        Regression Insights:
        {regression_insights}
        
        Generated Test Cases:
        {test_cases}
        
        Review criteria:
        1. Completeness - Do test cases cover all business rules from FSD?
        2. Coverage - Are all KPIs and report logic tested?
        3. Alignment - Do test cases align with regression insights?
        4. Quality - Are test cases clear, actionable, and well-structured?
        5. Traceability - Can each test case be traced back to requirements?
        6. Feasibility - Are test cases executable with available resources?
        
        Provide:
        - Final reviewed and refined test cases
        - Coverage analysis showing mapping to business rules
        - Recommendations for test execution priority
        - Any gaps or additional test cases needed
        """,
        agent=qa_reviewer,
        expected_output="Final reviewed test cases with coverage analysis and execution recommendations"
    )
    
    crew = Crew(
        agents=[qa_reviewer],
        tasks=[review_task],
        process="sequential"
    )
    
    result = crew.kickoff()
    reviewed = result if isinstance(result, str) else str(result)
    
    return {
        **state,
        "final_test_cases": reviewed
    }
