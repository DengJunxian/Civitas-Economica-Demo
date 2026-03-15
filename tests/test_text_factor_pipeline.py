from unittest.mock import MagicMock

from data_flywheel.text_factor_pipeline import TextFactorPipeline
from data_flywheel.nlp_processor import NlpProcessor
from data_flywheel.schemas import RawArticle


def test_text_factor_pipeline_keyword_mode():
    pipeline = TextFactorPipeline(enable_finbert=False, enable_bertopic=False)
    result = pipeline.analyze(
        title="央行宣布降准并释放流动性",
        content="银行板块受益，市场风险偏好回升。",
        llm_payload={},
    )

    assert "financial_factors" in result
    assert "dominant_topic" in result
    assert "topic_signals" in result
    assert result["dominant_topic"] != ""
    assert -1.0 <= float(result["sentiment_score"]) <= 1.0


def test_nlp_processor_fallback_contains_text_factors():
    mock_router = MagicMock()
    mock_router.sync_call_with_fallback.return_value = ("invalid json", "", "mock")

    processor = NlpProcessor(model_router=mock_router)
    article = RawArticle(
        title="监管处罚引发市场担忧",
        content="多家公司被问询，风险情绪升温。",
        source="unit-test",
    )
    event = processor.process(article)

    assert isinstance(event.text_factors, dict)
    assert "financial_factors" in event.text_factors
    financial = event.text_factors["financial_factors"]
    assert "panic_index" in financial
    assert "policy_shock" in financial

