from dataclasses import dataclass


@dataclass
class InvestigationPlan:
    question: str
    subquestions: list[str]


class InvestigationPlanner:
    """
    Creates an evidence-seeking investigation plan.

    The first version uses deterministic planning.
    Gemini-based planning will be added after the workflow
    is proven to work.
    """

    def create_plan(self, question: str) -> InvestigationPlan:
        question_lower = question.lower()

        subquestions = []

        if "why" in question_lower:
            subquestions.append(
                "What historical event or decision originally established "
                "the condition being questioned?"
            )

        if any(
            term in question_lower
            for term in ["after", "since", "still", "retained", "remain"]
        ):
            subquestions.append(
                "What changed after the original decision, including "
                "equipment, process, validation, or configuration changes?"
            )

            subquestions.append(
                "Were there later attempts to change, remove, or reaffirm "
                "the condition?"
            )

        if any(
            term in question_lower
            for term in ["technical", "reason", "justification", "rationale"]
        ):
            subquestions.append(
                "What documented technical evidence supports the current "
                "condition?"
            )

        subquestions.append(
            "What later records, decisions, or events could confirm, "
            "contradict, or qualify the historical explanation?"
        )

        return InvestigationPlan(
            question=question,
            subquestions=subquestions,
        )