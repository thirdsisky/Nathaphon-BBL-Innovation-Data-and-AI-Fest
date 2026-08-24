import re
import sys
import textwrap

from dotenv import load_dotenv
from src.agents import build_agent_graph

# Load the API key.
load_dotenv()

# Windows terminals default to a legacy codepage that cannot encode the
# curly quotes / en-dashes the LLM writes ("employee's" -> "employee?s").
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WRAP_WIDTH = 160

SAMPLE_QUERIES = [
    "What is the policy on international travel?",
    "How many days a week can employees work remotely?",
    "Tell me about cats and how they sleep.",
    "What receipts do I need for expenses on an international trip?",
    "Can I get reimbursed for a home office desk and chair?",
    "How is my data and my devices kept secure when I work from home?",
    "What is a dog"
]

#Pull similarity scores out so they can be printed on their own line."
def split_scores(snippets: str) -> tuple[str, list[str]]:
    scores = re.findall(r"\[similarity ([\d.]+)\]\n", snippets)
    text = re.sub(r"\[similarity [\d.]+\]\n", "", snippets)
    return text, scores

#Wrap long lines to WRAP_WIDTH so the output stays readable in a terminal
def wrap(text: str) -> str:
    wrapped = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped:
            wrapped.append(line)
            continue
        indent = line[: len(line) - len(stripped)]
        hanging = indent + "  " if stripped.startswith(("-", "*")) else indent
        wrapped.append(
            textwrap.fill(
                stripped,
                width=WRAP_WIDTH,
                initial_indent=indent,
                subsequent_indent=hanging,
            )
        )
    return "\n".join(wrapped)


def main():
    queries = sys.argv[1:] or SAMPLE_QUERIES
    app = build_agent_graph()

    for query in queries:
        result = app.invoke(
            {
                "user_query": query,
                "retrieved_snippets": "",
                "final_report": "",
            }
        )

        snippets, scores = split_scores(result["retrieved_snippets"])

        print(f"\nQuery: {query}")
        print(f"\nRetrieved:\n{wrap(snippets)}")
        print(f"\nSimilarity: {', '.join(scores) if scores else 'no match'}")
        print(f"\nAnswer:\n{wrap(result['final_report'])}")
        print()


if __name__ == "__main__":
    main()
