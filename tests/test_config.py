import pytest

from unpangram.config import Config, ConfigError, load_config, require_env


def test_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg == Config()
    assert cfg.rotation == [
        "moonshotai/kimi-k2.5",
        "deepseek/deepseek-v4.1-flash",
        "qwen/qwen3.8-flash",
    ]
    assert (cfg.max_rounds, cfg.similarity_threshold) == (2, 0.8)
    assert cfg.generate_model == "anthropic/claude-sonnet-4.6"
    assert cfg.pangram_model == "pangram-4"


def test_yaml_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("max_rounds: 4\nrotation: [a/b, c/d]\nmax_usd: 0.5\n")
    cfg = load_config(str(path))
    assert (cfg.max_rounds, cfg.rotation, cfg.max_usd) == (4, ["a/b", "c/d"], 0.5)


def test_config_path_from_env(tmp_path, monkeypatch):
    path = tmp_path / "c.yaml"
    path.write_text("similarity_threshold: 0.9\n")
    monkeypatch.setenv("UNPANGRAM_CONFIG", str(path))
    assert load_config().similarity_threshold == 0.9


def test_unknown_key_and_missing_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("nope: 1\n")
    with pytest.raises(ConfigError, match="nope"):
        load_config(str(path))
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "missing.yaml"))


def test_require_env(monkeypatch):
    monkeypatch.delenv("PANGRAM_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="PANGRAM_API_KEY"):
        require_env("PANGRAM_API_KEY")
