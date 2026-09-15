import json

import pytest

from app.investigation.planner import InvestigationPlanner


class FakePlanningClient:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_planner_rejects_empty_question():
    planner = InvestigationPlanner()

    with pytest.raises(ValueError):
        planner.create_plan("")


def test_planner_rejects_whitespace_question():
    planner = InvestigationPlanner()

    with pytest.raises(ValueError):
        planner.create_plan("   ")


def test_deterministic_planner_preserves_question():
    planner = InvestigationPlanner()

    question = "Why is this limit 42.3°C?"

    plan = planner.create_plan(question)

    assert plan.question == question


def test_gemini_planner_uses_generated_subquestions():
    response = json.dumps(
        {
            "subquestions": [
                "Where did the limit originate?",
                "What technical evidence justified the original limit?",
                "What changed after the equipment was replaced?",
                "Was the limit later reviewed or changed?",
                "What evidence supports the limit today?",
            ]
        }
    )

    client = FakePlanningClient(response)
    planner = InvestigationPlanner(planning_client=client)

    plan = planner.create_plan("Why is this limit 42.3°C?")

    assert len(plan.subquestions) == 7
    assert plan.subquestions[0] == "Where did the limit originate?"
    assert len(client.prompts) == 1


def test_gemini_prompt_contains_question_and_planning_rules():
    client = FakePlanningClient(
        json.dumps(
            {
                "subquestions": [
                    "Where did the condition originate?",
                    "What evidence justified it?",
                    "What changed afterward?",
                ]
            }
        )
    )

    planner = InvestigationPlanner(planning_client=client)

    planner.create_plan("Why is this limit 42.3°C?")

    prompt = client.prompts[0]

    assert "Why is this limit 42.3°C?" in prompt
    assert "Do not answer the user's question" in prompt
    assert "Do not invent facts" in prompt
    assert "evidence-seeking" in prompt


def test_gemini_duplicate_subquestions_are_removed():
    response = json.dumps(
        {
            "subquestions": [
                "Where did the condition originate?",
                "Where did the condition originate?",
                "What evidence justified it?",
                "What changed afterward?",
            ]
        }
    )

    client = FakePlanningClient(response)
    planner = InvestigationPlanner(planning_client=client)

    plan = planner.create_plan("Explain this condition.")

    assert len(plan.subquestions) == 3


def test_invalid_gemini_json_falls_back_to_deterministic_planner():
    client = FakePlanningClient("not valid json")

    planner = InvestigationPlanner(planning_client=client)

    plan = planner.create_plan("Why is this limit 42.3°C?")

    assert plan.question == "Why is this limit 42.3°C?"
    assert len(plan.subquestions) >= 3


def test_invalid_gemini_structure_falls_back():
    client = FakePlanningClient(
        json.dumps(
            {
                "wrong_field": [
                    "Where did it originate?",
                ]
            }
        )
    )

    planner = InvestigationPlanner(planning_client=client)

    plan = planner.create_plan("Why is this limit 42.3°C?")

    assert len(plan.subquestions) >= 3


def test_too_few_gemini_questions_falls_back():
    client = FakePlanningClient(
        json.dumps(
            {
                "subquestions": [
                    "Where did it originate?",
                ]
            }
        )
    )

    planner = InvestigationPlanner(planning_client=client)

    plan = planner.create_plan("Why is this limit 42.3°C?")

    assert len(plan.subquestions) >= 3


def test_too_many_gemini_questions_falls_back():
    client = FakePlanningClient(
        json.dumps(
            {
                "subquestions": [
                    "Question 1",
                    "Question 2",
                    "Question 3",
                    "Question 4",
                    "Question 5",
                    "Question 6",
                    "Question 7",
                    "Question 8",
                    "Question 9",
                ]
            }
        )
    )

    planner = InvestigationPlanner(planning_client=client)

    plan = planner.create_plan("Why is this limit 42.3°C?")

    assert len(plan.subquestions) >= 3


def test_non_string_gemini_questions_are_ignored():
    client = FakePlanningClient(
        json.dumps(
            {
                "subquestions": [
                    "Where did it originate?",
                    123,
                    "What evidence justified it?",
                    "What changed afterward?",
                ]
            }
        )
    )

    planner = InvestigationPlanner(planning_client=client)

    plan = planner.create_plan("Explain this condition.")

    assert len(plan.subquestions) == 3
    assert 123 not in plan.subquestions


def test_no_gemini_client_uses_deterministic_planner():
    planner = InvestigationPlanner()

    plan = planner.create_plan(
        "Why is this limit still retained today?"
    )

    assert len(plan.subquestions) >= 3
    assert any(
        "originally established" in question
        for question in plan.subquestions
    )


def test_gemini_planner_adds_required_investigation_anchors():
    response = json.dumps(
        {
            "subquestions": [
                "Where did the limit originate?",
                "What technical evidence justified the original limit?",
                "What changed after the equipment was replaced?",
                "Was the limit later reviewed or changed?",
                "What evidence supports the limit today?",
                "What evidence is missing?",
            ]
        }
    )

    client = FakePlanningClient(response)
    planner = InvestigationPlanner(planning_client=client)

    plan = planner.create_plan("Why is this limit 42.3°C?")

    assert (
        "Was a proposal to change this parameter ever submitted, "
        "and if so, what was the stated reason for its approval "
        "or rejection?"
    ) in plan.subquestions

    assert (
        "Did the equipment-qualification or requalification "
        "documentation explicitly state what it does and does "
        "not establish about the current operating limit?"
    ) in plan.subquestions

    assert len(plan.subquestions) == 8