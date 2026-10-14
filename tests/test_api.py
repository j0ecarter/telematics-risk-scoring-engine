import asyncio

import httpx

from apexrisk.api.main import app


def test_health_endpoint_identifies_synthetic_mode() -> None:
    async def request_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/health")

    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "apexrisk",
        "version": "0.1.0",
        "data_mode": "synthetic",
        "production_pricing": False,
    }
