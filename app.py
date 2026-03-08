import sys
import os
import logging

# Configure secure logging (preventing sensitive info from leaking to standard out)
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='system_errors.log'
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.rag_pipeline import HellobooksRAG

# Security Configuration
MAX_QUERY_LENGTH = 500  # Prevent DoS via massive input string evaluation

def main():
    print("=====================================================")
    print("        Hellobooks AI - Accounting Assistant         ")
    print("        [Secure Production Build]                    ")
    print("=====================================================")
    print("Type 'exit' or 'quit' to close the application.")

    try:
        rag_system = HellobooksRAG()
        print("\n[System] Ready to answer your accounting questions.")
    except Exception as e:
        logging.error(f"Failed initialization: {e}")
        print("\n[Error] Failed to initialize RAG pipeline. Please contact the administrator.")
        return

    while True:
        try:
            # 1. Capture and strip user input securely
            question = input("\n[You] Ask an accounting question: ").strip()

            if not question:
                continue

            # 2. Input Validation & Sanitization (Defending against resource exhaustion)
            if len(question) > MAX_QUERY_LENGTH:
                print(f"\n[Warning] Input exceeds the {MAX_QUERY_LENGTH} character limit. Please ask a more concise question.")
                continue

            if question.lower() in ['exit', 'quit']:
                print("\nGoodbye! Thank you for using Hellobooks AI.")
                break

            print("\n[Hellobooks AI] Thinking...")

            # 3. Safe Execution
            answer = rag_system.answer_question(question)
            
            # The pipeline now guarantees string responses securely
            print(f"\n[Hellobooks AI]\n{answer}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            # 4. Safe Error Handling: Do not leak stack traces or internal structure
            logging.error(f"Runtime error during query '{question[:50]}...': {e}")
            print("\n[Error] An unexpected error occurred while processing your request. Please try again later.")

if __name__ == "__main__":
    main()
