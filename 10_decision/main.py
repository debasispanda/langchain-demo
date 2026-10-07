import time

from dotenv import load_dotenv
from langchain_typesafe import Choice, Noul, Score, Question, TypeSafeClassifier

load_dotenv()

classifier = TypeSafeClassifier(
    base_url="http://localhost:11434", model="tev1", api_key="ollama"
)

STATE = "The deploy failed twice and customers are seeing 500s. Can someone look now?"

QUESTIONS: dict[str, Question] = {
    "urgent": Noul(instructions="Does this need attention right now?"),
    "severity": Score(
        instructions="How severe is the impact?",
        criteria=["Cosmetic.", "Degraded for some users.", "Full outage."],
    ),
    "team": Choice(
        instructions="Which team should handle this?",
        criteria={
            "infra": "Deploys, availability, and on-call incidents.",
            "billing": "Payments, invoices, and subscriptions.",
            "support": "Customer support and helpdesk related issues.",
        },
    ),
}

start = time.perf_counter()
response = classifier.invoke({"state": STATE, "questions": QUESTIONS})
exe_time = (time.perf_counter() - start) * 1000  # Convert to milliseconds

model = response.model

urgent = response.nouls["urgent"].noul

severity = response.scores["severity"].score

team = response.choices["team"].choice
confidence = response.choices["team"].confidence

print(f"Model: {model}")
print(f"Execution Time: {exe_time:.0f} ms")

print(f"Urgent: {urgent:.2f} (Noul)")
print(f"Severity: {severity:.2f} (Score)")
print(f"Team: {team} (Choice), Confidence: {confidence:.2f}")
