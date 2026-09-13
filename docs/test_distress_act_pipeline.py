from pathlib import Path

from smoke_rag_pipeline import run

if __name__ == "__main__":
    run(
        pdf=Path(__file__).with_name("Distress-Act-1951.pdf"),
        title="Distress Act 1951",
        question="Which goods are exempt from distress proceedings?",
        follow_up="How is the warrant executed?",
    )
