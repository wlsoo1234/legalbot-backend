from pathlib import Path

from smoke_rag_pipeline import run

if __name__ == "__main__":
    run(
        pdf=Path(__file__).with_name("SPECIFIC-RELIEF-ACT-1950.pdf"),
        title="Specific Relief Act 1950",
        question="When may a court grant specific performance in Malaysia?",
        follow_up="What about a perpetual injunction?",
    )
