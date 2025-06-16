import argparse

from src.core import Core

def main(args):
    core = Core(config_path=args.config, conversational=True)
    qa_chain = core.qa_chain
    history = []

    while True:
        question = input("Ask a question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break

        result = qa_chain.invoke({"question": question, "chat_history": history})
        answer = result['answer']

        print("Answer:", answer)

        history.append((question, answer))
        if len(history) > 5:
            history = history[-5:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chat CLI")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args)
