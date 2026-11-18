from fastapi.testclient import TestClient

from assistente_docs.api.app import create_app
from assistente_docs.api.ratelimit import RateLimiter
from assistente_docs.config import Settings
from fakes import FakeAskService
from test_api import ANSWER


class Clock:
    def __init__(self, now: float = 1_763_500_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_each_visitor_has_its_own_hourly_window() -> None:
    clock = Clock()
    limiter = RateLimiter(per_hour=2, per_day=0, clock=clock)

    assert limiter.check("a") is None
    assert limiter.check("a") is None
    assert "2 perguntas por hora" in limiter.check("a")
    assert limiter.check("b") is None

    clock.now += 3600
    assert limiter.check("a") is None


def test_the_daily_total_counts_every_visitor_and_resets_the_next_day() -> None:
    clock = Clock()
    limiter = RateLimiter(per_hour=0, per_day=2, clock=clock)

    assert limiter.check("a") is None
    assert limiter.check("b") is None
    assert "limite de perguntas de hoje" in limiter.check("c")

    clock.now += 86_400
    assert limiter.check("c") is None


def test_refused_requests_do_not_count_against_the_daily_total() -> None:
    limiter = RateLimiter(per_hour=1, per_day=2, clock=Clock())

    assert limiter.check("a") is None
    assert limiter.check("a") is not None  # refused by the hourly limit
    assert limiter.check("b") is None  # still room in the day


def test_zero_means_no_limit() -> None:
    limiter = RateLimiter(per_hour=0, per_day=0, clock=Clock())

    assert all(limiter.check("a") is None for _ in range(100))


def demo_client(per_hour: int) -> TestClient:
    settings = Settings(_env_file=None, demo_questions_per_hour=per_hour)
    return TestClient(create_app(FakeAskService(ANSWER), settings=settings))


def test_the_api_answers_429_with_a_readable_message() -> None:
    with demo_client(per_hour=1) as client:
        first = client.post("/ask", json={"question": "Como me autentico?"})
        second = client.post("/ask", json={"question": "Como me autentico?"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "por hora" in second.json()["detail"]


def test_visitors_behind_the_tunnel_are_told_apart_by_cf_connecting_ip() -> None:
    with demo_client(per_hour=1) as client:
        ana = client.post(
            "/ask", json={"question": "Pix?"}, headers={"CF-Connecting-IP": "1.1.1.1"}
        )
        bia = client.post(
            "/ask", json={"question": "Pix?"}, headers={"CF-Connecting-IP": "2.2.2.2"}
        )
        ana_again = client.post(
            "/ask", json={"question": "Pix?"}, headers={"CF-Connecting-IP": "1.1.1.1"}
        )

    assert (ana.status_code, bia.status_code, ana_again.status_code) == (200, 200, 429)


def test_the_demo_page_is_served_at_the_root() -> None:
    with demo_client(per_hour=0) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Assistente da API do Asaas" in response.text
    assert 'fetch("ask"' in response.text
