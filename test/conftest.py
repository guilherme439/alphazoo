import os

import pytest
import ray


@pytest.fixture(scope="session", autouse=True)
def ray_session():
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False
    )
    yield
    ray.shutdown()


@pytest.fixture
def work_dir(tmp_path):
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_dir)
