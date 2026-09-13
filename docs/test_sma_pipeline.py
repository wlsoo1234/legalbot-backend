from pathlib import Path

from smoke_rag_pipeline import run

if __name__ == "__main__":
    run(
        pdf=Path(__file__).with_name("Strata-Management-Act-2013.pdf"),
        title="Strata Management Act 2013",
        question="What are the duties of a management corporation?",
        follow_up="What powers does the Commissioner of Buildings have?",
    )
