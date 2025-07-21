import os
import json
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

class SimpleRAGTestGenerator:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
    def load_documents(self, folder_path: str):
        """Load all documents from a folder"""
        docs = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.endswith('.pdf'):
                        docs.extend(PyPDFLoader(file_path).load())
                    elif file.endswith('.csv'):
                        docs.extend(CSVLoader(file_path).load())
                    elif file.endswith(('.txt', '.py', '.sql')):
                        docs.extend(TextLoader(file_path, encoding='utf-8').load())
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        return self.text_splitter.split_documents(docs)
    
    def create_vectorstore(self, documents):
        """Create vector store from documents"""
        return Chroma.from_documents(documents, self.embeddings)
    
    def generate_test_cases(self, vectorstore, query: str):
        """Generate test cases based on query"""
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
        
        test_case_prompt = f"""Based on the provided documents, {query}
        
        Generate test cases with:
        1. Test Case ID
        2. Description
        3. Steps
        4. Expected Result
        
        Be specific and practical."""
        
        result = qa_chain({"query": test_case_prompt})
        return {
            "test_cases": result["result"],
            "sources": [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        }

def main():
    # Setup
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    generator = SimpleRAGTestGenerator(OPENAI_API_KEY)
    
    # Load documents
    print("Loading documents...")
    documents = generator.load_documents("./documents")  # Put all your documents here
    print(f"Loaded {len(documents)} document chunks")
    
    # Create vector store
    print("Creating vector store...")
    vectorstore = generator.create_vectorstore(documents)
    
    # Generate test cases examples
    queries = [
        "generate test cases for user authentication based on the business requirements",
        "create test cases for the Tableau sales dashboard focusing on data accuracy",
        "generate integration test cases for the API endpoints shown in the code",
        "create test cases for data validation based on the business rules"
    ]
    
    results = {}
    for query in queries:
        print(f"\nGenerating: {query}")
        result = generator.generate_test_cases(vectorstore, query)
        results[query] = result
        print(f"Generated test cases from {len(result['sources'])} sources")
    
    # Save results
    with open("test_cases_output.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Interactive mode
    print("\n--- Interactive Mode ---")
    while True:
        user_query = input("\nEnter test case request (or 'exit'): ")
        if user_query.lower() == 'exit':
            break
        
        result = generator.generate_test_cases(vectorstore, user_query)
        print("\nGenerated Test Cases:")
        print(result["test_cases"])
        print(f"\nSources: {result['sources']}")

if __name__ == "__main__":
    main()
