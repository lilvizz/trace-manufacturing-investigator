from dataclasses import dataclass
from enum import Enum


class DemoOutcome(str, Enum):
    FLAG = "flag"
    CLEAR = "clear"
    NOT_ESTABLISHED = "not_established"


@dataclass(frozen=True)
class InvestigationScenario:
    id: str
    title: str
    question: str
    outcome: DemoOutcome
    outcome_label: str
    outcome_description: str
    evidence_files: tuple[str, ...]
    why_it_matters: str


SCENARIOS: tuple[InvestigationScenario, ...] = (
    InvestigationScenario(
        id="press07_42_3_basis",
        title="Press-07 · 42.3°C technical basis",
        question=(
            "Why is Press-07 limited to 42.3°C, what changed since that "
            "limit was established, and is its current technical basis still supported?"
        ),
        outcome=DemoOutcome.NOT_ESTABLISHED,
        outcome_label="NOT ESTABLISHED",
        outcome_description=(
            "Historical and qualification evidence exists, but the corpus does not "
            "establish an explicit current technical reapproval of the exact 42.3°C limit."
        ),
        evidence_files=(
            "07_Thermal_Validation_2001.txt",
            "08_Thermal_Excursion_Investigation_2001.txt",
            "04_ECO_2011_184_Press_Replacement.txt",
            "13_Process_Requalification_2014.txt",
            "15_SPC_Current_Configuration.csv",
            "25_SOP_Cure_Process_v8_Current.txt",
        ),
        why_it_matters="Hero case: demonstrates evidence-gap handling without overclaiming.",
    ),
    InvestigationScenario(
        id="rtd_7714_2_calibration_failure",
        title="RTD-7714-2 · calibration failure",
        question=(
            "Did the 2018 RTD-7714-2 calibration failure represent a measurement-integrity "
            "issue requiring corrective action?"
        ),
        outcome=DemoOutcome.FLAG,
        outcome_label="FLAG",
        outcome_description=(
            "The sensor recorded a 0.31°C as-found deviation against a 0.30°C acceptance "
            "threshold, failed calibration, and was replaced."
        ),
        evidence_files=(
            "26_Temperature_Sensor_Calibration_History.csv",
            "28_Temperature_Sensor_Replacement_2018.txt",
            "29_Calibration_Procedure_v4.txt",
        ),
        why_it_matters="Shows TRACE can surface a concrete documented failure requiring action.",
    ),
    InvestigationScenario(
        id="rtd_7714_2_process_control",
        title="RTD replacement · process-control impact",
        question=(
            "Did the 2018 RTD-7714-2 replacement indicate a broader Press-07 process-control problem?"
        ),
        outcome=DemoOutcome.CLEAR,
        outcome_label="NO ESCALATION",
        outcome_description=(
            "The replacement was like-for-like; three post-replacement batches were within normal "
            "quality parameters, and the work order explicitly records no process-control issue."
        ),
        evidence_files=(
            "28_Temperature_Sensor_Replacement_2018.txt",
            "26_Temperature_Sensor_Calibration_History.csv",
        ),
        why_it_matters="Directly fixes Q48 by demonstrating a correct non-flag outcome.",
    ),
    InvestigationScenario(
        id="rejected_43c_change",
        title="43.0°C proposal · why was the change rejected?",
        question=(
            "Why was the proposed increase of the Press-07 UCL from 42.3°C to 43.0°C not adopted in 2014?"
        ),
        outcome=DemoOutcome.CLEAR,
        outcome_label="NO TECHNICAL FAILURE",
        outcome_description=(
            "The proposal was rejected for customer re-approval timing and commercial scheduling, "
            "not because the characterization data technically failed."
        ),
        evidence_files=(
            "38_Rejected_Proposal_2014_43C.txt",
            "14_Engineering_Change_2014_117.txt",
        ),
        why_it_matters="Shows TRACE can separate technical evidence from administrative decisions.",
    ),
)


def get_scenarios() -> tuple[InvestigationScenario, ...]:
    return SCENARIOS


def get_scenario(scenario_id: str) -> InvestigationScenario:
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(f"Unknown investigation scenario: {scenario_id}")
