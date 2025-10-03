"""
Contract tests for /api/v1/explain endpoint
"""
import pytest
from pydantic import ValidationError

from app.core.contracts import ExplainResponse


class TestExplainContract:
    """Test explain endpoint contract"""

    def test_explain_response_valid_structure(self):
        """Test valid explain response"""
        response_data = {
            "symbol": "AAPL",
            "plain_english": "We recommend buying AAPL because...",
            "glossary": [
                {"term": "Support", "definition": "Price level where buying interest appears"}
            ],
            "decision_tree": [
                "1. AI model predicted 2.3% expected return",
                "2. Price holding above support level"
            ],
            "confidence_breakdown": {
                "overall_score": 0.75,
                "quality_score": 0.70,
                "confidence_value": 0.85
            },
            "risk_factors": [
                "Stock has higher than average volatility"
            ]
        }

        explain = ExplainResponse(**response_data)
        assert explain.symbol == "AAPL"
        assert len(explain.glossary) == 1
        assert len(explain.decision_tree) == 2

    def test_explain_response_missing_required_field(self):
        """Test explain response with missing field"""
        response_data = {
            # Missing symbol
            "plain_english": "Test",
            "glossary": [],
            "decision_tree": [],
            "confidence_breakdown": {},
            "risk_factors": []
        }

        with pytest.raises(ValidationError):
            ExplainResponse(**response_data)

    def test_glossary_structure(self):
        """Test glossary item structure"""
        response_data = {
            "symbol": "AAPL",
            "plain_english": "Test",
            "glossary": [
                {"term": "RSI", "definition": "Relative Strength Index"},
                {"term": "MACD", "definition": "Moving Average Convergence Divergence"}
            ],
            "decision_tree": [],
            "confidence_breakdown": {},
            "risk_factors": []
        }

        explain = ExplainResponse(**response_data)
        assert len(explain.glossary) == 2
        assert explain.glossary[0]["term"] == "RSI"
