from crewai import Agent, Task, Crew

# Your existing agent definition
fsd_reader = Agent(
    name="FSD Reader",
    role="Functional Specification Analyst",
    goal="Extract detailed business rules, data definitions, and validation logic from FSD documents",
    backstory="Expert in interpreting functional specs to derive actionable business rules and data requirements",
    llm=client,
    verbose=True,  # Optional: for debugging
    allow_delegation=False  # Optional: prevent delegation to other agents
)

def fsd_node(state):
    """
    LangGraph node function that processes FSD documents using CrewAI agent
    """
    # Extract text from the FSD document
    fsd_text = extract_text_from_docx(state["fsd_path"])
    
    # Create a task for the agent
    fsd_analysis_task = Task(
        description=f"""
        Analyze the following Functional Specification Document and extract:
        1. Detailed business rules and logic
        2. Data definitions and structures  
        3. Validation rules and constraints
        4. Process workflows and dependencies
        5. Integration requirements
        
        FSD Content:
        {fsd_text}
        
        Please provide a structured analysis with clear categorization of each type of rule.
        """,
        agent=fsd_reader,
        expected_output="Structured analysis containing business rules, data definitions, validation logic, and process workflows extracted from the FSD document"
    )
    
    # Create a crew with the agent and task
    fsd_crew = Crew(
        agents=[fsd_reader],
        tasks=[fsd_analysis_task],
        verbose=True,  # Optional: for debugging
        process="sequential"  # Since we only have one task
    )
    
    # Execute the crew and get results
    try:
        result = fsd_crew.kickoff()
        rules = result if isinstance(result, str) else str(result)
    except Exception as e:
        print(f"Error processing FSD with CrewAI: {e}")
        rules = f"Error processing FSD: {str(e)}"
    
    # Return updated state
    return {**state, "fsd_rules": rules}

# Alternative simpler approach if you don't need full Crew functionality
def fsd_node_simple(state):
    """
    Simplified version using agent directly (if your CrewAI version supports it)
    """
    fsd_text = extract_text_from_docx(state["fsd_path"])
    
    # Create task description
    task_description = f"""
    Extract detailed business rules, data definitions, and validation logic from this FSD:
    
    {fsd_text}
    
    Please structure your response with clear sections for:
    - Business Rules
    - Data Definitions  
    - Validation Logic
    - Process Workflows
    """
    
    try:
        # Some CrewAI versions allow direct agent execution
        rules = fsd_reader.execute_task(task_description)
    except AttributeError:
        # Fallback to crew-based approach
        task = Task(
            description=task_description,
            agent=fsd_reader,
            expected_output="Structured FSD analysis"
        )
        
        crew = Crew(
            agents=[fsd_reader],
            tasks=[task],
            process="sequential"
        )
        
        result = crew.kickoff()
        rules = result if isinstance(result, str) else str(result)
    
    return {**state, "fsd_rules": rules}
