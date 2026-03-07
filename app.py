import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.rag_pipeline import HellobooksRAG


def main():
    print("=====================================================")
    print("        Hellobooks AI - Accounting Assistant         ")
    print("=====================================================")
    print("Type 'exit' or 'quit' to close the application.")

    try:
        rag_system = HellobooksRAG()
        print("\n[System] Ready to answer your accounting questions.")
    except Exception as e:
        print(f"\n[Error] Failed to initialize RAG pipeline:\n{e}")
        return

    while True:
        try:
            question = input("\n[You] Ask an accounting question: ").strip()

            if not question:
                continue

            if question.lower() in ['exit', 'quit']:
                print("\nGoodbye! Thank you for using Hellobooks AI.")
                break

            print("\n[Hellobooks AI] Thinking...")

            answer = rag_system.answer_question(question)

            print(f"\n[Hellobooks AI]\n{answer}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
