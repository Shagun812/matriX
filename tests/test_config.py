import pytest

from matrix.agents import AgentConfig
from matrix.config import freeze, load_frozen, verify
from matrix.schemas import CHECKS


@pytest.fixture
def inputs(tmp_path):
    table_path = tmp_path / "table.npz"
    table_path.write_bytes(b"temporary likelihood table")

    prior_path = tmp_path / "prior.pkl"
    prior_path.write_bytes(b"temporary prior model")

    cfg = AgentConfig(
        costs={c: 1.0 for c in CHECKS},
        budget=None,
        eps_gain=0.02,
        prior_eps=0.05,
        k_samples=3,
        seed=17,
    )

    return tmp_path, table_path, prior_path, cfg


def freeze_test_config(tmp_path, table_path, prior_path, cfg):
    return freeze(
        path=tmp_path / "frozen.json",
        cfg=cfg,
        table_path=table_path,
        prior_path=prior_path,
        prompt_version="test-v1",
        tau_grid=[0.6, 0.7, 0.8, 0.9],
        operating_tau=0.8,
        llm_model="fake-model",
        llm_params={"temperature": 0},
        notes="Unit test only",
    )


def test_freeze_and_load_round_trip(inputs):
    tmp_path, table_path, prior_path, cfg = inputs

    original = freeze_test_config(
        tmp_path, table_path, prior_path, cfg
    )
    loaded = load_frozen(tmp_path / "frozen.json")

    assert loaded == original
    assert loaded.agent_config() == cfg
    verify(
        loaded,
        cfg,
        table_path,
        prior_path,
        prompt_version="test-v1",
        llm_model="fake-model",
        llm_params={"temperature": 0},
    )


def test_freeze_refuses_to_overwrite_existing_file(inputs):
    tmp_path, table_path, prior_path, cfg = inputs
    freeze_test_config(tmp_path, table_path, prior_path, cfg)

    with pytest.raises(FileExistsError):
        freeze_test_config(tmp_path, table_path, prior_path, cfg)


def test_verify_detects_changed_likelihood_table(inputs):
    tmp_path, table_path, prior_path, cfg = inputs
    frozen = freeze_test_config(tmp_path, table_path, prior_path, cfg)

    table_path.write_bytes(b"modified likelihood table")

    with pytest.raises(ValueError, match="likelihood table differs"):
        verify(
            frozen, cfg, table_path, prior_path,
            prompt_version="test-v1",
        )


def test_verify_detects_changed_prior_model(inputs):
    tmp_path, table_path, prior_path, cfg = inputs
    frozen = freeze_test_config(tmp_path, table_path, prior_path, cfg)

    prior_path.write_bytes(b"modified prior model")

    with pytest.raises(ValueError, match="prior model differs"):
        verify(
            frozen, cfg, table_path, prior_path,
            prompt_version="test-v1",
        )


def test_verify_detects_prompt_version_change(inputs):
    tmp_path, table_path, prior_path, cfg = inputs
    frozen = freeze_test_config(tmp_path, table_path, prior_path, cfg)

    with pytest.raises(ValueError, match="prompt version"):
        verify(
            frozen, cfg, table_path, prior_path,
            prompt_version="test-v2",
        )


def test_verify_detects_agent_config_change(inputs):
    tmp_path, table_path, prior_path, cfg = inputs
    frozen = freeze_test_config(tmp_path, table_path, prior_path, cfg)

    changed_cfg = AgentConfig(
        costs=dict(cfg.costs),
        budget=cfg.budget,
        eps_gain=0.03,
        prior_eps=cfg.prior_eps,
        k_samples=cfg.k_samples,
        seed=cfg.seed,
    )

    with pytest.raises(ValueError, match="agent config differs"):
        verify(
            frozen, changed_cfg, table_path, prior_path,
            prompt_version="test-v1",
        )


def test_exclusive_creation_preserves_existing_file(inputs):
    tmp_path, table_path, prior_path, cfg = inputs
    destination = tmp_path / "frozen.json"
    destination.write_text("preserve this file", encoding="utf-8")

    with pytest.raises(FileExistsError):
        freeze(
            path=destination,
            cfg=cfg,
            table_path=table_path,
            prior_path=prior_path,
            prompt_version="test-v1",
            tau_grid=[0.6, 0.7, 0.8, 0.9],
            operating_tau=0.8,
        )

    assert destination.read_text(encoding="utf-8") == "preserve this file"


def test_verify_detects_changed_llm_parameters(inputs):
    tmp_path, table_path, prior_path, cfg = inputs
    frozen = freeze_test_config(tmp_path, table_path, prior_path, cfg)

    with pytest.raises(ValueError, match="llm params differ"):
        verify(
            frozen,
            cfg,
            table_path,
            prior_path,
            prompt_version="test-v1",
            llm_model="fake-model",
            llm_params={"temperature": 0.5},
        )
