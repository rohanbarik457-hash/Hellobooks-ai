import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.rag_pipeline import HellobooksRAG

def run_tests():
    print("Beginning 10 Pipeline Integrity Tests...")
    
    try:
        rag = HellobooksRAG()
    except Exception as e:
        print(f"FAILED on initialization: {e}")
        return False
        
    queries = [
        "what is an invoice",
        "tell me about bookkeeping",
        "how does cash flow work",
        "what is a balance sheet",
        "define profit and loss",
        "invoice kya hai",
        "book keeping definitions",
        "what is remote work", # tests dynamic bounds
        "ceo?",
        "what is unearned revenue"
    ]
    
    success_count = 0
    for i, q in enumerate(queries, 1):
        print(f"Test {i}/10: '{q}'...")
        try:
            answer = rag.answer_question(q)
            if not answer or "Error" in answer:
                print(f"Test {i} FAILED: Returned invalid answer - {answer}")
                return False
            success_count += 1
            print(f"Test {i} PASSED.")
        except Exception as e:
            print(f"Test {i} FAILED with Exception: {e}")
            return False
            
    if success_count == 10:
        print("\nSUCCESS: All 10 tests passed with zero errors.")
        return True
    return False

if __name__ == "__main__":
    if not run_tests():
        sys.exit(1)
    sys.exit(0)
