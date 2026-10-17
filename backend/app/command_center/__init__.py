"""TRACE Investigation Command Center — multi-page presentation layer.

Presentation only. Every page reads the InvestigationResult produced by the
existing pipeline (app.investigation.investigator.Investigator.run) and the
helpers in app.ui. Nothing in this package plans, retrieves, reasons,
verifies or calibrates; it never fabricates findings, citations,
relationships or confidence values.
"""
