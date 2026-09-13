from app.investigation.planner import InvestigationPlanner


def test_planner_creates_investigation_plan():
    planner = InvestigationPlanner()

    question = (
        "Why was the 42.3 degree limit retained "
        "after the 2014 requalification?"
    )

    plan = planner.create_plan(question)

    assert plan.question == question
    assert len(plan.subquestions) >= 3

    combined = " ".join(plan.subquestions).lower()

    assert "established" in combined
    assert "changed" in combined
    assert "attempts" in combined