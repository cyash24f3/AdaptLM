import pytest
from fastapi.testclient import TestClient

from adaptlm.api.app import create_app
from adaptlm.config import Settings


@pytest.fixture
def settings():
    return Settings(profile="fixture", admin_token="test-credential", _env_file=None)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client
