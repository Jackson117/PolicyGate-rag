"""supervise.py against fake runners: a stalled one is dumped, stopped and resumed."""

import sys

from policygate_rag import supervise


def fake_runner(tmp_path, body):
    script = tmp_path / "fake_runner.py"
    script.write_text(body)
    return script


def test_a_stalled_runner_is_restarted_with_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(supervise, "ROOT", tmp_path)
    calls = tmp_path / "calls.txt"
    # First invocation hangs without writing; the second (with --resume) succeeds.
    script = fake_runner(tmp_path, f"""
import sys, time, pathlib
pathlib.Path({str(calls)!r}).open("a").write(" ".join(sys.argv[1:]) + "\\n")
out = pathlib.Path({str(tmp_path)!r}) / "run"
out.mkdir(exist_ok=True); (out / "plan.json").write_text("{{}}")
if "--resume" not in sys.argv:
    time.sleep(60)
""")
    monkeypatch.setattr(supervise.subprocess, "Popen",
                        lambda cmd, **kw: _popen([sys.executable, str(script), *cmd[3:]], **kw))
    code = supervise.supervise(["--out", "run"], tmp_path / "log.txt",
                               stall_s=1, max_restarts=2, poll_s=0.5)
    assert code == 0
    lines = calls.read_text().splitlines()
    assert len(lines) == 2 and "--resume" not in lines[0] and "--resume" in lines[1]
    out = capsys.readouterr().out
    assert "STALL" in out and "restart 1/2" in out and "done" in out


def test_it_gives_up_after_max_restarts(tmp_path, monkeypatch):
    monkeypatch.setattr(supervise, "ROOT", tmp_path)
    script = fake_runner(tmp_path, "import sys; sys.exit(3)")
    monkeypatch.setattr(supervise.subprocess, "Popen",
                        lambda cmd, **kw: _popen([sys.executable, str(script)], **kw))
    assert supervise.supervise(["--out", "run"], tmp_path / "log.txt",
                               stall_s=60, max_restarts=1, poll_s=0.1) == 3


_real_popen = supervise.subprocess.Popen


def _popen(cmd, **kw):
    kw.pop("cwd", None)
    return _real_popen(cmd, **kw)
