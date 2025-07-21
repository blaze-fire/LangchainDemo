import os
import json
from typing import List, Dict, Tuple
from datetime import datetime
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class OptimizedRAGTestGenerator:
    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model, temperature=0.3)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", ";", ",", " ", ""]
        )
        self.vectorstores = {}
        
    def load_documents_from_folders(self, folder_paths: Dict[str, str]) -> Dict[str, List]:
        """Load documents from separate folders with intelligent parsing"""
        documents = {}
        
        # Define file handlers for each extension
        file_handlers = {
            ".pdf": lambda path: PyPDFLoader(path).load(),
            ".csv": lambda path: CSVLoader(path).load(),
            ".xlsx": lambda path: UnstructuredExcelLoader(path).load(),
            ".xls": lambda path: UnstructuredExcelLoader(path).load(),
            ".txt": lambda path: TextLoader(path, encoding='utf-8').load(),
            ".py": lambda path: TextLoader(path, encoding='utf-8').load(),
            ".sql": lambda path: TextLoader(path, encoding='utf-8').load(),
            ".js": lambda path: TextLoader(path, encoding='utf-8').load(),
            ".java": lambda path: TextLoader(path, encoding='utf-8').load()
        }
        
        for doc_type, folder_path in folder_paths.items():
            documents[doc_type] = []
            
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} does not exist")
                continue
                
            # Process all files in the folder
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    
                    if ext in file_handlers:
                        try:
                            # Load documents
                            docs = file_handlers[ext](file_path)
                            
                            # Enhance metadata
                            for doc in docs:
                                doc.metadata.update({
                                    "doc_type": doc_type,
                                    "file_name": file,
                                    "file_path": file_path,
                                    "extension": ext
                                })
                            
                            documents[doc_type].extend(docs)
                            
                        except Exception as e:
                            print(f"Error loading {file}: {e}")
        
        # Apply intelligent text splitting
        for doc_type in documents:
            if documents[doc_type]:
                # Use different chunk sizes for different document types
                chunk_size = 2000 if doc_type == "code" else 1500
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=300,
                    separators=["\n\n", "\n", ".", ";", ",", " ", ""]
                )
                documents[doc_type] = splitter.split_documents(documents[doc_type])
                print(f"Loaded {len(documents[doc_type])} chunks from {doc_type}")
                
        return documents
    
    def create_optimized_vectorstores(self, documents: Dict[str, List]):
        """Create separate and combined vector stores with optimized settings"""
        # Create individual vector stores for focused retrieval
        for doc_type, docs in documents.items():
            if docs:
                self.vectorstores[doc_type] = Chroma.from_documents(
                    docs, 
                    self.embeddings,
                    collection_name=doc_type,
                    persist_directory=f"./chroma_db/{doc_type}"
                )
        
        # Create combined vectorstore for cross-reference queries
        all_docs = []
        for docs in documents.values():
            all_docs.extend(docs)
        
        if all_docs:
            self.vectorstores["combined"] = Chroma.from_documents(
                all_docs,
                self.embeddings,
                collection_name="combined",
                persist_directory="./chroma_db/combined"
            )
    
    def generate_test_cases(self, query: str, doc_type: str = "combined", use_context: bool = True) -> Dict:
        """Generate high-quality test cases with optimized retrieval"""
        
        # Context-aware prompts for different document types
        prompts = {
            "business_docs": """You are an expert QA engineer analyzing business requirements and documentation.

Context from business documents:
{context}

Based on the above context, {question}

Generate comprehensive test cases with the following structure:
1. Test Case ID: TC_BUS_XXX
2. Test Scenario: Clear description of what is being tested
3. Business Rule/Requirement: Reference to specific business rule
4. Preconditions: Initial state and setup required
5. Test Steps: Detailed numbered steps
6. Expected Results: Specific outcomes with business validation
7. Test Data: Sample data requirements
8. Priority: High/Medium/Low based on business impact

Focus on business logic validation, user workflows, data integrity, and edge cases.""",

            "tableau_reports": """You are a BI testing specialist focused on Tableau report validation.

Context from Tableau reports and data:
{context}

Based on the above context, {question}

Generate data-centric test cases with:
1. Test Case ID: TC_TAB_XXX
2. Report/Dashboard Name: Specific Tableau component
3. Data Validation Checks: Specific calculations and aggregations
4. Filter Test Scenarios: All filter combinations
5. Performance Metrics: Load time, refresh rate targets
6. Visual Validation: Layout, formatting, responsiveness
7. Export Tests: PDF, Excel, Data extract validation
8. Cross-Reference: Comparison with source data

Include SQL queries or data validation formulas where applicable.""",

            "code": """You are a senior test engineer analyzing source code and technical implementation.

Context from code files:
{context}

Based on the above context, {question}

Generate technical test cases including:
1. Test Case ID: TC_CODE_XXX
2. Component/Function: Specific code element being tested
3. Test Type: Unit/Integration/API/Performance
4. Test Setup: Environment and dependencies
5. Test Implementation: Actual test code or pseudocode
6. Assertions: Specific validation checks
7. Edge Cases: Boundary conditions, error scenarios
8. Mocking Requirements: External dependencies

Include code snippets and specific test frameworks where relevant.""",

            "combined": """You are a comprehensive QA architect analyzing the entire system.

Context from all documentation:
{context}

Based on the above context, {question}

Generate end-to-end test cases covering:
1. Test Case ID: TC_E2E_XXX
2. Test Objective: High-level goal
3. System Components: All involved modules
4. Integration Points: APIs, databases, UI elements
5. Test Flow: Complete user journey with data flow
6. Validation Points: Business rules, data accuracy, UI state
7. Non-Functional Tests: Performance, security, usability
8. Rollback Scenarios: Error handling and recovery

Consider cross-system dependencies and data consistency."""
        }
        
        # Get appropriate retriever
        if doc_type not in self.vectorstores:
            return {"error": f"No documents loaded for type: {doc_type}"}
        
        # Use MMR for diverse results and increase k for better context
        retriever = self.vectorstores[doc_type].as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8 if use_context else 4,
                "fetch_k": 20,
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )
        
        # Create QA chain with optimized prompt
        prompt = PromptTemplate(
            template=prompts.get(doc_type, prompts["combined"]),
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": False
            }
        )
        
        # Generate test cases
        result = qa_chain({"query": query})
        
        # Process and structure the response
        return {
            "test_cases": result["result"],
            "metadata": {
                "doc_type": doc_type,
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "sources_count": len(result["source_documents"])
            },
            "sources": [
                {
                    "file": doc.metadata.get("file_name", "Unknown"),
                    "type": doc.metadata.get("doc_type", "Unknown"),
                    "path": doc.metadata.get("file_path", ""),
                    "preview": doc.page_content[:150].replace("\n", " ") + "..."
                }
                for doc in result["source_documents"]
            ]
        }
    
    def batch_generate_optimized(self, test_requests: List[Tuple[str, str, str]]) -> Dict:
        """Batch generate test cases with progress tracking"""
        results = {}
        
        for i, (name, query, doc_type) in enumerate(test_requests, 1):
            print(f"\n[{i}/{len(test_requests)}] Generating: {name}")
            results[name] = self.generate_test_cases(query, doc_type)
            
            # Quick summary
            sources_count = results[name]["metadata"]["sources_count"]
            print(f"✓ Generated test cases using {sources_count} source documents")
            
        return results

def main():
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize generator
    generator = OptimizedRAGTestGenerator(OPENAI_API_KEY, model="gpt-4")
    
    # Define your folder structure
    folder_paths = {
        "business_docs": "./business_documents",     # Business requirements, specs
        "tableau_reports": "./tableau_reports",      # Tableau exports, CSVs
        "code": "./source_code"                     # Python, SQL, etc.
    }
    
    # Load documents from separate folders
    print("Loading documents from separate folders...")
    documents = generator.load_documents_from_folders(folder_paths)
    
    if not any(documents.values()):
        print("\nNo documents found! Please add documents to these folders:")
        for doc_type, path in folder_paths.items():
            print(f"  - {path}/ (for {doc_type})")
        return
    
    # Create optimized vector stores
    print("\nCreating optimized vector stores...")
    generator.create_optimized_vectorstores(documents)
    
    # High-value test scenarios
    test_requests = [
        # Business-focused tests
        ("User Authentication Tests", 
         "Generate comprehensive test cases for user login, registration, and password reset functionality",
         "business_docs"),
        
        # Tableau-focused tests
        ("Sales Dashboard Validation", 
         "Create test cases to validate all calculations, filters, and data accuracy in the sales dashboard",
         "tableau_reports"),
        
        # Code-focused tests
        ("API Endpoint Tests", 
         "Generate unit and integration tests for all REST API endpoints including error handling",
         "code"),
        
        # Cross-functional tests
        ("End-to-End Order Processing", 
         "Create test cases for complete order flow from UI submission through payment processing to data storage",
         "combined"),
        
        ("Data Consistency Tests",
         "Generate test cases to verify data consistency between application, database, and Tableau reports",
         "combined")
    ]
    
    # Generate test cases
    print("\n" + "="*60)
    print("GENERATING TEST CASES")
    print("="*60)
    
    results = generator.batch_generate_optimized(test_requests)
    
    # Save results with timestamp
    output_file = f"test_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("\nDocument types available:")
    for doc_type in generator.vectorstores.keys():
        doc_count = len(documents.get(doc_type, []))
        print(f"  - {doc_type} ({doc_count} chunks)")
    
    print("\nEnter 'exit' to quit")
    
    while True:
        print("\n" + "-"*40)
        query = input("Test case request: ").strip()
        
        if query.lower() == 'exit':
            break
        
        if not query:
            continue
            
        # Show available document types
        print("\nSelect document type:")
        print("  1. business_docs (Business requirements)")
        print("  2. tableau_reports (BI/Analytics)")
        print("  3. code (Source code)")
        print("  4. combined (All documents)")
        
        choice = input("Enter choice (1-4, default=4): ").strip() or "4"
        
        doc_type_map = {
            "1": "business_docs",
            "2": "tableau_reports", 
            "3": "code",
            "4": "combined"
        }
        
        doc_type = doc_type_map.get(choice, "combined")
        
        print(f"\nGenerating test cases from {doc_type}...")
        result = generator.generate_test_cases(query, doc_type)
        
        # Display results
        print("\n" + "="*60)
        print("GENERATED TEST CASES")
        print("="*60)
        print(result["test_cases"])
        
        print(f"\n📚 Sources Used: {result['metadata']['sources_count']}")
        for src in result["sources"][:3]:  # Show first 3 sources
            print(f"  - {src['file']} ({src['type']})")

if __name__ == "__main__":
    main()
