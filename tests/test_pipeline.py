import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coverage_summary import ArticleAnalysis, analyse_mock, dedupe, rollup  # noqa: E402


def art(title, text="x"):
    return {"title": title, "text": text, "source": "S"}


def test_dedupe_removes_syndicated_copy():
    articles = [art("SkyJet adds Lisbon route to growing network"), art("SkyJet adds daily Lisbon route to growing network")]
    assert len(dedupe(articles)) == 1


def test_dedupe_keeps_different_stories():
    articles = [art("SkyJet strike delays flights"), art("SkyJet wins industry award")]
    assert len(dedupe(articles)) == 2


def test_rollup_counts_are_exact():
    a = lambda s, t: ArticleAnalysis(sentiment=s, theme=t, summary="s", evidence_quote="q")
    analyses = [(art("a"), a("negative", "strike")), (art("b"), a("negative", "strike")), (art("c"), a("positive", "route"))]
    result = rollup(analyses)
    assert result["total"] == 3
    assert result["sentiments"] == {"negative": 2, "positive": 1}
    assert result["read_first"]["title"] == "a"


def test_mock_flags_strike_as_negative():
    result = analyse_mock(art("Strike delays flights", "Flights cancelled."), "SkyJet")
    assert result.sentiment == "negative"
