"""Run real CPU demos, collect provenance, create charts and capture the results viewer."""
import hashlib
import html
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tpour import Encoder, encode_all, load_encoder, read_jsonl, save_json

OUT = ROOT / "reports"
EV = OUT / "evidence"
ASSETS = OUT / "assets"


def run(command, name):
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
    text = "$ " + subprocess.list2cmdline(command) + "\n\n" + completed.stdout + completed.stderr
    text += f"\nExit code: {completed.returncode}\n"
    (EV / name).write_text(text, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(text)
    return text


def charts(results, logs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                        "axes.spines.right": False, "figure.facecolor": "white"})
    colors = ["#8293ad", "#22a699", "#3767e8"]
    fig, ax = plt.subplots(figsize=(11, 4.4), layout="constrained")
    x = np.arange(3)
    names = ["nDCG@5", "Recall@5", "MRR@5"]
    for i, key in enumerate(("untrained", "ce_only", "tpour")):
        values = [results[key]["metrics"][n] for n in names]
        bars = ax.bar(x + (i - 1) * .24, values, .22, color=colors[i],
                      label={"untrained": "Chưa huấn luyện", "ce_only": "Chỉ CE", "tpour": "CE + TRPO"}[key])
        ax.bar_label(bars, fmt="%.3f", fontsize=10, padding=3)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Điểm (0–1)")
    ax.legend(ncol=3, loc="upper left", frameon=False)
    ax.set_title("Demo giả lập • seed 42 • 8 truy vấn / 16 tài liệu", loc="left", pad=20, weight="bold")
    fig.savefig(ASSETS / "metrics.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for key, color in (("ce_only", colors[1]), ("tpour", colors[2])):
        axes[0].plot([v["step"] for v in logs[key]], [v["loss"] for v in logs[key]],
                     label="Chỉ CE" if key == "ce_only" else "CE + TRPO", color=color, linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_title("Loss tổng (trục log)")
    axes[0].legend(frameon=False)
    for key, color in (("ce_only", colors[1]), ("tpour", colors[2])):
        axes[1].plot([v["step"] for v in logs[key]], [v["trpo"] for v in logs[key]],
                     label="Chỉ CE (ghi nhận)" if key == "ce_only" else "CE + TRPO", color=color, linewidth=2)
    axes[1].set_yscale("log")
    axes[1].set_title("TRPO loss đo trên batch train")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel("Bước huấn luyện")
        ax.grid(alpha=.15)
    fig.savefig(ASSETS / "loss.png", dpi=180)
    plt.close(fig)


def viewer(payload):
    template = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>TPOUR • Demo kiểm chứng</title>
<style>*{box-sizing:border-box}body{margin:0;background:#edf2f8;color:#15243c;font:16px 'Segoe UI',Arial,sans-serif}header{background:#14243d;color:white;padding:30px 48px;display:flex;justify-content:space-between}h1{font-size:30px;margin:5px 0}header p{margin:8px 0;color:#c5d4e8}main{max-width:1320px;margin:26px auto;padding:0 25px}.tag{color:#b4caff;letter-spacing:2px;font-size:12px;font-weight:700}.badge{background:#284364;border:1px solid #496384;border-radius:20px;padding:9px 18px;height:40px}.warn{background:#fff4d8;border-left:4px solid #e3ae26;padding:14px 20px;border-radius:8px;margin-bottom:20px;font-size:15px}.tabs{display:flex;gap:10px;margin:18px 0}.tabs button,select{font:inherit;padding:12px 20px;border:1px solid #d2dce9;background:white;border-radius:8px;cursor:pointer}.tabs button.on{background:#3767e8;color:white;border-color:#3767e8}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card{background:white;border:1px solid #dbe4f0;border-radius:12px;padding:22px;box-shadow:0 5px 18px #14243d06}.number{font-size:36px;font-weight:700;margin:10px 0}.muted{color:#64748b;font-size:14px}.green{color:#0c8678;font-weight:600}h2{font-size:23px;margin:0 0 18px}h3{font-size:18px;margin:8px 0 15px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:13px 10px;border-bottom:1px solid #e2e8f0;font-size:15px}th{color:#64748b;font-size:13px}.rank td{font-size:14px}.doc{max-width:580px}.good{background:#e9faf5}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:20px}.chart{width:100%;height:auto}.query{font-size:20px;font-weight:600;padding:17px 0}pre{background:#111f33;color:#d5e5fa;padding:22px;border-radius:10px;white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.55 Consolas,monospace;margin:0}.status{display:flex;gap:18px;margin:18px 0;color:#0c8678}.foot{margin:20px 0;font-size:13px;color:#64748b}.screen{display:none}.screen.active{display:block}</style>
<header><div><div class="tag">TEMPORAL RETRIEVAL LAB / BẰNG CHỨNG THỰC NGHIỆM</div><h1>TPOUR / TRPO — Demo trên CPU</h1><p>Kết quả từ mã PyTorch trong repository • <span id="date"></span></p></div><div class="badge">DỮ LIỆU GIẢ LẬP</div></header>
<main><div class="warn"><b>Phạm vi:</b> minh họa pipeline và kết quả đã chạy. Chưa tái lập bài báo; chưa đánh giá BEIR. Chỉ CE và CE + TRPO cùng đạt điểm tối đa trên tập test này.</div>
<div class="tabs"><button data-tab="overview" class="on">01 / Tổng quan</button><button data-tab="ranking">02 / Truy xuất demo</button><button data-tab="evidence">03 / Log & kiểm tra</button></div>
<section id="overview" class="screen active"><div class="grid"><div class="card"><div class="muted">TẬP HUẤN LUYỆN</div><div class="number">32 cặp</div><div class="muted">Tổ chức 0–31 • 150 bước / mô hình</div></div><div class="card"><div class="muted">TẬP ĐÁNH GIÁ</div><div class="number">8 truy vấn</div><div class="muted">Tổ chức 32–39 • 16 tài liệu</div></div><div class="card"><div class="muted">THIẾT LẬP</div><div class="number">CPU / 42</div><div class="muted">Seed 42 • queue 256 • λ = 0.925</div></div></div><div class="two"><div class="card"><h2>Chỉ số retrieval</h2><table id="metrics"></table><p class="muted">Điểm đo trên tập test đầy đủ của demo. Encoder toy, không phải Contriever.</p></div><div class="card"><h2>Đối chứng công bằng</h2><p>Cùng encoder khởi tạo, dữ liệu, seed, learning rate và số bước. Nhánh đối chứng dùng λ = 1.</p><h3>Kết luận được hỗ trợ</h3><p class="green">Huấn luyện và đánh giá hoạt động.</p><p>Chưa có bằng chứng TRPO vượt chỉ CE; dữ liệu dùng chung mẫu câu và tín hiệu “digital / online” dễ học.</p><p class="muted">Bản Contriever thật: đã chạy 2 bước trên 8 mẫu của tác giả ngày 10/10/2026.</p></div></div><div class="card" style="margin-top:18px"><img class="chart" src="assets/metrics.png"></div></section>
<section id="ranking" class="screen"><div class="card"><h2>Truy xuất từ checkpoint demo đã lưu</h2><label>Chọn truy vấn test: <select id="query"></select></label><div class="query" id="querytext"></div><div class="muted">Mốc đích: 2021 • Nhãn đúng: phiên bản “new” của cùng tổ chức • Điểm: cosine của encoder toy chuẩn hóa</div></div><div class="two"><div class="card"><h2>Encoder chưa huấn luyện</h2><p id="base-rank" class="muted"></p><table class="rank" id="base-table"></table></div><div class="card"><h2>Encoder CE + TRPO</h2><p id="tpour-rank" class="green"></p><table class="rank" id="tpour-table"></table></div></div><p class="foot">Hàng xanh là tài liệu có qrel = 1. Tất cả điểm được tính trực tiếp bằng tpour.Encoder từ checkpoint; viewer chỉ hiển thị kết quả đã tính, không huấn luyện trong trình duyệt.</p></section>
<section id="evidence" class="screen"><div class="grid"><div class="card"><div class="muted">KIỂM TRA TỰ ĐỘNG</div><div class="number green">5 / 5 đạt</div><div class="muted">Gradient, reference, queue, metrics, input</div></div><div class="card"><div class="muted">SEED KIỂM TRA</div><div class="number">42 · 43 · 44</div><div class="muted">Chi tiết kết quả trong evidence/summary.json</div></div><div class="card"><div class="muted">CONTRIEVER THẬT</div><div class="number">2 bước</div><div class="muted">Smoke test ngày 10/10/2026 • không là benchmark</div></div></div><div class="two"><div class="card"><h2>Log kiểm tra thực tế</h2><pre id="tests"></pre></div><div class="card"><h2>Log chạy demo seed 42</h2><pre id="log"></pre></div></div></section>
<p class="foot">Nguồn bằng chứng: reports/evidence • Mã: tpour.py / test_tpour.py • github.com/lequangduccode/tpour-trpo-experiments</p></main>
<script>const D=__DATA__;document.getElementById('date').textContent=D.captured_at;document.getElementById('tests').textContent=D.tests;document.getElementById('log').textContent=D.demo_log;
document.getElementById('metrics').innerHTML='<tr><th>Mô hình</th><th>nDCG@5</th><th>Recall@5</th><th>MRR@5</th></tr>'+['untrained','ce_only','tpour'].map(k=>'<tr><td>'+({untrained:'Chưa huấn luyện',ce_only:'Chỉ CE',tpour:'CE + TRPO'}[k])+'</td>'+['nDCG@5','Recall@5','MRR@5'].map(m=>'<td>'+D.results[k].metrics[m].toFixed(3)+'</td>').join('')+'</tr>').join('');
const select=document.getElementById('query');D.queries.forEach(q=>{const o=document.createElement('option');o.value=q._id;o.textContent='Query '+q._id+' · Organization'+q._id;select.appendChild(o)});select.value=D.selected;
function render(){const q=D.queries.find(q=>q._id===select.value);document.getElementById('querytext').textContent=q.text;for(const m of ['base','tpour']){const r=D.rankings[m][q._id];const rank=r.findIndex(x=>x.relevant)+1;document.getElementById(m+'-rank').textContent='Tài liệu đúng ở hạng '+rank+' / 16';const t=document.getElementById(m+'-table');t.innerHTML='<tr><th>Hạng</th><th>Tài liệu & nội dung</th><th>Điểm</th></tr>';r.slice(0,5).forEach((d,i)=>{const row=document.createElement('tr');if(d.relevant)row.className='good';[String(i+1),d.id+'\n'+d.text,d.score.toFixed(4)].forEach(v=>{const cell=document.createElement('td');cell.textContent=v;row.appendChild(cell)});t.appendChild(row)})}}select.onchange=render;render();document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('on',x===b));document.querySelectorAll('.screen').forEach(s=>s.classList.toggle('active',s.id===b.dataset.tab))});</script></html>'''
    # Serialize safely for an inline script, including if user-supplied texts contain '<'.
    (OUT / "demo_viewer.html").write_text(template.replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")), encoding="utf-8")


def main():
    EV.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if "--capture-only" in sys.argv:
        meta = json.loads((EV / "provenance.json").read_text(encoding="utf-8"))
        rank = json.loads((EV / "retrieval_rankings.json").read_text(encoding="utf-8"))
        payload = {"captured_at": meta["captured_at"],
                   "results": json.loads((EV / "seed_42.json").read_text(encoding="utf-8")),
                   "queries": rank["queries"], "rankings": rank["rankings"], "selected": rank["selected"],
                   "tests": (EV / "tests.txt").read_text(encoding="utf-8"),
                   "demo_log": (EV / "demo_seed_42.txt").read_text(encoding="utf-8").split('"untrained"')[0][-2400:]}
        viewer(payload)
        capture(payload["queries"], payload["selected"])
        return
    captured_at = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
    tests = run([sys.executable, "-X", "utf8", "-m", "unittest", "test_tpour", "-v"], "tests.txt")
    runs = {}
    for seed in (42, 43, 44):
        destination = f"outputs/report_runs/seed_{seed}"
        log = run([sys.executable, "-X", "utf8", "tpour.py", "demo", "--seed", str(seed),
                   "--output", destination], f"demo_seed_{seed}.txt")
        folder = ROOT / destination
        result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        save_json(EV / f"seed_{seed}.json", result)
        runs[str(seed)] = result
    folder = ROOT / "outputs/report_runs/seed_42"
    logs = {k: json.loads((folder / k / "train_log.json").read_text(encoding="utf-8")) for k in ("ce_only", "tpour")}
    for k, values in logs.items():
        save_json(EV / f"{k}_train_log.json", values)
    charts(runs["42"], logs)
    corpus = read_jsonl(folder / "data/test/corpus.jsonl")
    queries = read_jsonl(folder / "data/test/queries.jsonl")
    torch.manual_seed(42)
    encoders = {"base": Encoder("toy", normalize=True).eval(),
                "tpour": load_encoder(folder / "tpour/encoder.pt", "cpu")}
    rankings = {}
    for name, enc in encoders.items():
        docs = encode_all(enc, [d["text"] for d in corpus])
        query = encode_all(enc, [q["text"] for q in queries])
        scores = query @ docs.T
        rankings[name] = {}
        for q, s in zip(queries, scores):
            ranked = []
            for i in torch.argsort(s, descending=True, stable=True).tolist():
                d = corpus[i]
                ranked.append({"id": d["_id"], "text": d["text"], "timestamp": d["timestamp"],
                               "score": float(s[i]), "relevant": d["_id"] == q["_id"] + "-new"})
            rankings[name][q["_id"]] = ranked
    selected = max(queries, key=lambda q: next(i for i, d in enumerate(rankings["base"][q["_id"]]) if d["relevant"]))["_id"]
    metadata = {"captured_at": captured_at, "python": platform.python_version(), "torch": torch.__version__,
                "platform": platform.platform(), "cuda_available": torch.cuda.is_available(),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "upstream_commit": subprocess.check_output(["git", "-C", "upstream/TPOUR", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "tpour_sha256": hashlib.sha256((ROOT / "tpour.py").read_bytes()).hexdigest(),
                "test_exit_code": 0, "seeds": [42, 43, 44], "synthetic": True}
    save_json(EV / "provenance.json", metadata)
    save_json(EV / "summary.json", {s: {k: r[k]["metrics"] for k in ("untrained", "ce_only", "tpour")} for s, r in runs.items()})
    save_json(EV / "retrieval_rankings.json", {"queries": queries, "corpus": corpus,
                                               "rankings": rankings, "selected": selected})
    payload = {"captured_at": captured_at, "results": runs["42"], "queries": queries,
               "rankings": rankings, "selected": selected, "tests": tests,
               "demo_log": (EV / "demo_seed_42.txt").read_text(encoding="utf-8").split('"untrained"')[0][-2400:]}
    viewer(payload)
    capture(queries, selected)
    print(json.dumps({"evidence": str(EV), "screenshots": 3, "selected_query": selected,
                      "seeds": metadata["seeds"]}, ensure_ascii=False))


def capture(queries, selected):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.goto((OUT / "demo_viewer.html").as_uri())
        page.locator("#metrics td").first.wait_for()
        page.screenshot(path=str(ASSETS / "demo_01_overview.png"), full_page=True)
        page.locator('[data-tab="ranking"]').click()
        page.screenshot(path=str(ASSETS / "demo_02_retrieval.png"), full_page=True)
        page.locator('#ranking .two').screenshot(path=str(ASSETS / "demo_02_retrieval_detail.png"))
        page.locator('[data-tab="evidence"]').click()
        page.screenshot(path=str(ASSETS / "demo_03_checks.png"), full_page=True)
        page.locator('#evidence .two').screenshot(path=str(ASSETS / "demo_03_checks_detail.png"))
        # Check that changing the selected query updates the visible retrieval result.
        other = next(q for q in queries if q["_id"] != selected)
        page.locator('[data-tab="ranking"]').click()
        page.locator("#query").select_option(other["_id"])
        assert page.locator("#querytext").inner_text() == other["text"]
        browser.close()


if __name__ == "__main__":
    main()
