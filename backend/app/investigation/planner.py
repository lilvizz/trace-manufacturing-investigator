import json
from dataclasses import dataclass
from typing import Protocol


@dataclass
class InvestigationPlan:
    question: str
    subquestions: list[str]


class PlanningClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class InvestigationPlanner:
    """
    Creates an evidence-seeking investigation plan.

    If a planning client is supplied, Gemini can generate the plan.
    The deterministic planner remains available as a safe fallback.
    """

    def __init__(
        self,
        planning_client: PlanningClient | None = None,
    ):
        self.planning_client = planning_client

    def create_plan(self, question: str) -> InvestigationPlan:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        question = question.strip()

        # Narrow factual/document questions should not trigger the full
        # historical/current-basis investigation. This preserves the
        # original planner below while adding a small, deterministic
        # routing layer for focused lookups.
        if self._is_narrow_question(question):
            return self._create_narrow_plan(question)

        if self.planning_client is None:
            return self._create_deterministic_plan(question)

        try:
            return self._create_gemini_plan(question)
        except Exception:
            return self._create_deterministic_plan(question)


    @staticmethod
    def _is_narrow_question(question: str) -> bool:
        q = " ".join(question.lower().split())
        if not q.startswith(("what ", "which ", "who ", "when ", "where ", "how many ")):
            return False

        deep_terms = (
            "why", "rationale", "justification", "revalidated",
            "revalidation", "re-validated", "re-justif", "technical basis",
            "still supported", "current technical", "original basis",
            "what changed since", "history", "should", "recommend",
            "current limit", "current parameter", "prove", "contradict",
        )
        if any(term in q for term in deep_terms):
            return False

        return any(term in q for term in (
            "equipment", "replacement", "replaced", "installed", "change",
            "changed", "document", "documents", "record", "records",
            "evidence", "failure", "failed", "fault", "incident",
            "proposal", "approval", "calibration", "work order", "eco",
        ))

    def _create_narrow_plan(self, question: str) -> InvestigationPlan:
        q = question.lower()
        subquestions: list[str] = []

        def add(text: str) -> None:
            if text not in subquestions:
                subquestions.append(text)

        if any(term in q for term in (
            "document", "documents", "record", "records", "evidence",
        )):
            add("Which documents or records directly address the specific event, condition, decision, or failure in the user's question?")
            add("What facts about the specific event, condition, decision, or failure are explicitly supported by those records?")

        if any(term in q for term in (
            "equipment", "replacement", "replaced", "installed", "change", "changed", "eco",
        )):
            add("What equipment, asset, or configuration was present before the change and what was installed or changed afterward?")
            add("What documented hardware, control, sensor, process, or configuration changes resulted from the event?")

        if any(term in q for term in (
            "failure", "failed", "fault", "defect", "incident",
        )):
            add("What documented failure, fault, defect, or incident occurred, and when was it recorded?")
            add("What documented cause, impact, corrective action, or follow-up is associated with the failure?")

        if "approval" in q or "proposal" in q:
            add("Which record contains the approval, rejection, or decision, and what reason is explicitly documented?")

        # Keep the narrow plan genuinely narrow. Two focused questions are
        # enough for a direct lookup; a third provides corroboration when
        # the wording is ambiguous.
        if not subquestions:
            add("Which records directly answer the user's factual question?")
            add("What facts are explicitly supported by those records?")

        return InvestigationPlan(
            question=question,
            subquestions=subquestions[:4],
        )

    def _create_gemini_plan(self, question: str) -> InvestigationPlan:
        prompt = self._build_prompt(question)
        response = self.planning_client.generate(prompt)

        plan = self._parse_response(response, question)

        if not plan.subquestions:
            raise ValueError("Gemini returned an empty investigation plan")

        return self._add_required_anchors(plan, question)

    def _build_prompt(self, question: str) -> str:
        return f"""
You are the investigation planner for TRACE, a manufacturing
evidence investigation system.

Your job is NOT to answer the user's question.

Your job is to determine which evidence-seeking questions TRACE
should investigate before producing an answer.

User question:
{question}

Create a small set of focused investigative subquestions.

The investigation should consider, when relevant:

1. Origin
   - Where did the current condition, decision, or value originate?

2. Original technical basis
   - What evidence originally justified it?

3. Changes over time
   - What equipment, process, validation, configuration, or
     specification changes occurred afterward?

4. Later decisions
   - Was the condition reviewed, changed, reaffirmed, rejected,
     or simply carried forward?

5. Current applicability
   - What evidence supports or contradicts the condition today?

6. Evidence gaps
   - What important information is missing or insufficient?

Rules:
- Do not answer the user's question.
- Do not invent facts.
- Do not assume the conclusion.
- Make each subquestion evidence-seeking.
- Avoid redundant subquestions.
- Return 3 to 8 subquestions.
- Keep each subquestion concise.

Return ONLY valid JSON in this exact structure:

{{
  "subquestions": [
    "question 1",
    "question 2",
    "question 3"
  ]
}}
""".strip()

    def _parse_response(
        self,
        response: str,
        question: str,
    ) -> InvestigationPlan:
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Gemini returned an empty response")

        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini returned invalid JSON") from exc

        if not isinstance(payload, dict):
            raise ValueError("Gemini response must be a JSON object")

        raw_subquestions = payload.get("subquestions")

        if not isinstance(raw_subquestions, list):
            raise ValueError("subquestions must be a list")

        subquestions: list[str] = []

        for item in raw_subquestions:
            if not isinstance(item, str):
                continue

            cleaned = item.strip()

            if cleaned and cleaned not in subquestions:
                subquestions.append(cleaned)

        if not 3 <= len(subquestions) <= 8:
            raise ValueError(
                "Gemini must return between 3 and 8 valid subquestions"
            )

        return InvestigationPlan(
            question=question,
            subquestions=subquestions,
        )

    def _add_required_anchors(
        self,
        plan: InvestigationPlan,
        question: str,
    ) -> InvestigationPlan:
        """
        Add deterministic investigation anchors only for questions
        that concern the origin, justification, persistence, or
        current applicability of a condition or parameter.

        Generic planning questions remain unchanged so that the
        planner's normal parsing and deduplication behavior is
        preserved.
        """

        question_lower = question.lower()

        anchor_terms = [
            "why",
            "limit",
            "threshold",
            "parameter",
            "setting",
            "value",
            "rationale",
            "justification",
            "retained",
            "remain",
            "reaffirm",
            "established",
            "origin",
            "current",
        ]

        if not any(
            term in question_lower
            for term in anchor_terms
        ):
            return plan

        required_anchors = [
            (
                "Was a proposal to change this parameter ever submitted, "
                "and if so, what was the stated reason for its approval "
                "or rejection?"
            ),
            (
                "Did the equipment-qualification or requalification "
                "documentation explicitly state what it does and does "
                "not establish about the current operating limit?"
            ),
        ]

        subquestions = list(plan.subquestions)

        for anchor in required_anchors:
            if anchor not in subquestions:
                subquestions.append(anchor)

        return InvestigationPlan(
            question=plan.question,
            subquestions=subquestions,
        )

    def _create_deterministic_plan(
        self,
        question: str,
    ) -> InvestigationPlan:
        question_lower = question.lower()
        subquestions: list[str] = []

        def add(question_text: str) -> None:
            if question_text not in subquestions:
                subquestions.append(question_text)

        if any(
            term in question_lower
            for term in [
                "why",
                "origin",
                "original",
                "established",
                "introduced",
                "created",
                "set",
            ]
        ):
            add(
                "What historical event, investigation, experiment, "
                "or engineering decision originally established the "
                "condition being questioned?"
            )

            add(
                "What documented technical evidence or observed behavior "
                "was used to justify the original decision?"
            )

        if any(
            term in question_lower
            for term in [
                "after",
                "since",
                "still",
                "remain",
                "retained",
                "changed",
                "change",
                "current",
                "today",
                "now",
            ]
        ):
            add(
                "What changed after the original decision, including "
                "equipment, process, validation, configuration, "
                "specifications, or operating conditions?"
            )

            add(
                "How did the condition evolve over time, and which "
                "later records confirm when it changed or remained unchanged?"
            )

        if any(
            term in question_lower
            for term in [
                "still",
                "remain",
                "retained",
                "review",
                "reaffirm",
                "remove",
                "raise",
                "lower",
                "change",
                "current",
            ]
        ):
            add(
                "Were there later reviews, change requests, validation "
                "activities, or attempts to modify, remove, or reaffirm "
                "the condition?"
            )

            add(
                "What was the reason for each later decision, and was "
                "the decision technical, administrative, commercial, "
                "or otherwise motivated?"
            )

        if any(
            term in question_lower
            for term in [
                "technical",
                "reason",
                "justification",
                "rationale",
                "safe",
                "valid",
                "applicable",
                "appropriate",
                "current",
                "still",
            ]
        ):
            add(
                "What documented technical evidence supports the "
                "condition as it applies to the current equipment, "
                "process, or configuration?"
            )

            add(
                "Does later evidence support, contradict, or leave "
                "uncertain the original technical rationale?"
            )

        add(
            "What later records, decisions, or events could confirm, "
            "contradict, or qualify the historical explanation?"
        )

        add(
            "What important evidence is missing or insufficient to "
            "establish a definitive conclusion?"
        )

        return InvestigationPlan(
            question=question,
            subquestions=subquestions,
        )