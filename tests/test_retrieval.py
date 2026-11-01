import pytest

from rag.graph import facts_from_seed, link_entities
from rag.hybrid import get_retriever, reciprocal_rank_fusion


def test_rrf_rewards_items_ranked_high_in_both_lists():
    assert reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])[:2] in ([1, 2], [2, 1])
    assert reciprocal_rank_fusion([[5, 6], [6, 7]])[0] == 6


def test_graph_links_three_month_course_to_ultra_combo():
    facts = facts_from_seed(link_entities("Which pack should I buy for a 3 month course?"))
    assert any("Ultra Strength Combo" in f and "₹19,000" in f for f in facts)


def test_graph_concern_to_product():
    facts = facts_from_seed(link_entities("Is there anything for PCOD?"))
    assert any("Ms. Perfect Women Wellness is designed to support PCOD" in f for f in facts)


@pytest.fixture(scope="module")
def retriever():
    return get_retriever()


@pytest.mark.parametrize("query,expected_source", [
    ("Where is my order? It was supposed to arrive yesterday.", "shipping_policy.md"),
    ("return sealed tins refund", "refund_policy.md"),
    ("forgot password cannot log in", "account_and_login.md"),
    ("oil bottle arrived broken and leaking", "damaged_or_wrong_item.md"),
    ("why pay 300 advance for cash on delivery", "payment_and_cod.md"),
])
def test_top_passage_comes_from_right_document(retriever, query, expected_source):
    result = retriever.retrieve(query)
    assert expected_source in [p.source for p in result.passages[:2]]


def test_out_of_scope_question_finds_nothing(retriever):
    assert not retriever.retrieve("custom branded gift hampers for 500 employees of my company").found


def test_graph_linking_ignores_hyphens():
    assert "mr_ultra" in link_entities("Which pack for a 3-month course?").get("packages", [])
