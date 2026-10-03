from pathlib import Path

import duckdb
import pytest

import fake_ollama
import locus_hf as H

GOOD, NOGATE = "hf_good_" + "x" * 20, "hf_nogate_" + "x" * 20


@pytest.fixture()
def hf(monkeypatch, tmp_path):
    srv, url = fake_ollama.start_hf()
    fake_ollama.HFH.seen.clear()
    monkeypatch.setenv("HF_ENDPOINT", url)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))          # Windows home
    monkeypatch.delenv("HF_HOME", raising=False)
    for n in H.ENV_NAMES:
        monkeypatch.delenv(n, raising=False)
    yield url
    srv.shutdown()


def test_token_precedence_and_saving(hf, monkeypatch, tmp_path):
    assert H.find_token("") == ("", "")
    cli = tmp_path / ".cache" / "huggingface" / "token"
    cli.parent.mkdir(parents=True)
    cli.write_text("hf_from_cli_" + "y" * 20, encoding="utf-8")
    assert H.find_token("")[1] == "huggingface-cli login"
    H.save_token("hf_saved_" + "z" * 20)
    assert H.find_token("")[1] == "saved by this app" and cli.read_text(encoding="utf-8").startswith("hf_from_cli")   # CLI file untouched
    monkeypatch.setenv("HF_TOKEN", "hf_env_" + "e" * 20)
    assert H.find_token("")[1] == "environment (HF_TOKEN)"
    assert H.find_token("hf_typed_" + "t" * 20)[1] == "typed in app"
    assert H.forget_saved_token() and not H.own_token_path().exists() and H.find_token("")[1].startswith("environment")
    assert H.mask("hf_abcdefghijklmnop1234") == "hf_…1234" and H.looks_like_token(GOOD) and not H.looks_like_token("nope")


def test_whoami_and_dataset_access_states(hf):
    assert H.whoami(GOOD) == {"ok": True, "name": "alice", "error": ""}
    assert not H.whoami("bad")["ok"] and "401" in H.whoami("bad")["error"]
    assert H.dataset_access(GOOD)["state"] == "ok"                          # a 302 means authorised
    assert H.dataset_access(NOGATE)["state"] == "gated" and "accept" in H.dataset_access(NOGATE)["message"].lower()
    assert H.dataset_access("")["state"] == "no-token" and H.dataset_access("bad")["state"] == "bad-token"
    # the token went to the HF endpoint only, and the redirect target (a CDN) was never contacted
    assert all(p.startswith(("/api/whoami", "/datasets/")) for _, p, _ in fake_ollama.HFH.seen)
    assert any(m == "HEAD" for m, _, _ in fake_ollama.HFH.seen)


def test_unreachable_hf_is_reported(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:1")
    assert H.dataset_access("hf_x" + "x" * 20)["state"] == "network" and "reach" in H.whoami("hf_x" + "x" * 20)["error"].lower()


def test_duckdb_apply_and_probe_report_errors_not_exceptions(env):
    con = duckdb.connect()
    ok, msg = H.apply_to_duckdb(con, "it's a token")            # quote in the token must not break the SQL
    assert isinstance(ok, bool) and msg                              # offline sandbox: httpfs install fails -> reported, not raised
    pr = H.probe_duckdb(con.cursor(), env["template"], "va")
    assert pr["ok"] and pr["rows"] == 3                              # local file stands in for hf://
    bad = H.probe_duckdb(con.cursor(), env["template"], "zz")
    assert not bad["ok"] and "file not found" in bad["error"]


def test_app_token_panel(env, hf, monkeypatch):
    from streamlit.testing.v1 import AppTest
    app = str(Path(__file__).resolve().parents[1] / "locus_explorer.py")
    at = AppTest.from_file(app, default_timeout=120).run()
    assert not at.exception
    assert any("No token yet" in w.value for w in at.warning)
    at.text_input(key="hf_token_input").set_value(GOOD).run()
    at.button(key="hf_use").click().run()
    assert not at.exception
    assert any("typed in app" in s.value for s in at.success)               # now using the pasted token
    assert at.text_input(key="hf_token_input").value == ""                     # the secret is cleared from the box
    assert not H.own_token_path().exists()                                     # not saved unless asked
    at.button(key="hf_check").click().run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Signed in as" in md and "Access to vaquill/open-us-law confirmed" in md
    at.button(key="hf_forget").click().run()
    assert any("No token yet" in w.value for w in at.warning)
