from __future__ import annotations


def test_chat_rejects_oversized_query_before_work(client):
    response = client.post(
        "/api/v1/chat",
        json={"query": "x" * 4001},
    )

    assert response.status_code == 422
