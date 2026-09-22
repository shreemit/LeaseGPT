"""Render evaluation results as a self-contained static HTML dashboard."""

from html import escape
import json
from pathlib import Path


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _metric_card(label: str, value: str, delta: str = "") -> str:
    delta_html = f'<span class="delta">{escape(delta)}</span>' if delta else ""
    return (
        '<div class="card metric">'
        f"<span>{escape(label)}</span><strong>{escape(value)}</strong>{delta_html}"
        "</div>"
    )


def _retrieval_chart(results: dict, metric: str, title: str) -> str:
    summaries = results["retrieval_summary"]
    groups = []
    for k in results["metadata"]["ks"]:
        bars = []
        for config in results["metadata"]["configs"]:
            value = summaries[config][f"{metric}@{k}"]
            bars.append(
                f'<div class="bar-wrap" title="{escape(config)}: {_percent(value)}">'
                f'<div class="bar {escape(config)}" style="height:{value * 100:.3f}%"></div>'
                f'<span>{_percent(value)}</span></div>'
            )
        groups.append(
            '<div class="bar-group">'
            f'<div class="bars">{"".join(bars)}</div><strong>@{k}</strong>'
            "</div>"
        )
    return (
        '<section class="card chart-card">'
        f"<h3>{escape(title)}</h3>"
        f'<div class="chart">{"".join(groups)}</div>'
        '<div class="legend"><i class="baseline"></i>Baseline '
        '<i class="constraint_rerank"></i>Constraint re-rank</div>'
        "</section>"
    )


def _score_histogram(rows: list[dict], field: str, title: str) -> str:
    counts = {score: sum(row.get(field) == score for row in rows) for score in range(1, 6)}
    maximum = max(counts.values(), default=1) or 1
    bars = []
    for score, count in counts.items():
        bars.append(
            '<div class="bar-group">'
            '<div class="bars single">'
            f'<div class="bar score" style="height:{count / maximum * 100:.3f}%"></div>'
            f"<span>{count}</span></div><strong>{score}</strong></div>"
        )
    return (
        '<section class="card chart-card">'
        f"<h3>{escape(title)}</h3>"
        f'<div class="chart histogram">{"".join(bars)}</div>'
        '<div class="axis-note">Score (1–5)</div></section>'
    )


def _retrieval_rows(results: dict) -> str:
    rows = []
    for row in results["retrieval_queries"]:
        baseline = row["configs"]["baseline"]
        reranked = row["configs"]["constraint_rerank"]
        before = baseline["recall@5"]
        after = reranked["recall@5"]
        if before is None:
            impact = "unscored"
            impact_label = "Unscored hard case"
        elif after > before:
            impact = "helped"
            impact_label = "Helped"
        elif after < before:
            impact = "hurt"
            impact_label = "Hurt"
        else:
            impact = "same"
            impact_label = "No change"
        baseline_hits = "—" if baseline["hits@5"] is None else baseline["hits@5"]
        reranked_hits = "—" if reranked["hits@5"] is None else reranked["hits@5"]
        rows.append(
            f'<tr data-impact="{impact}">'
            f"<td>{escape(row['id'])}</td>"
            f"<td>{escape(row['query'])}</td>"
            f"<td>{escape(row.get('notes', ''))}</td>"
            f"<td>{len(row['relevant_ids'])}</td>"
            f"<td>{baseline_hits}</td><td>{reranked_hits}</td>"
            f'<td><span class="pill {impact}">{impact_label}</span></td></tr>'
        )
    return "".join(rows)


def _generation_rows(rows: list[dict]) -> str:
    rendered = []
    for row in rows:
        faithfulness = row.get("faithfulness")
        relevance = row.get("relevance")
        rationale = row.get("rationale") or row.get("error") or ""
        rendered.append(
            "<tr>"
            f"<td>{escape(row['id'])}</td><td>{escape(row['query'])}</td>"
            f"<td>{faithfulness if faithfulness is not None else '—'}</td>"
            f"<td>{relevance if relevance is not None else '—'}</td>"
            f"<td>{escape(rationale)}</td></tr>"
        )
    return "".join(rendered)


def render_html(results: dict) -> str:
    metadata = results["metadata"]
    baseline = results["retrieval_summary"]["baseline"]
    reranked = results["retrieval_summary"]["constraint_rerank"]
    generation = results["generation_summary"]
    generation_rows = results.get("generation_queries", [])
    scored_rows = [
        row for row in generation_rows if row.get("faithfulness") is not None
    ]
    retrieval_delta = reranked["recall@5"] - baseline["recall@5"]
    precision_delta = reranked["precision@5"] - baseline["precision@5"]

    if scored_rows:
        generation_section = f"""
        <section>
          <h2>Generation quality</h2>
          <div class="metrics">
            {_metric_card("Average faithfulness", f"{generation['faithfulness_average']:.2f}/5")}
            {_metric_card("Average relevance", f"{generation['relevance_average']:.2f}/5")}
            {_metric_card("Judged queries", str(generation['judged_query_count']))}
          </div>
          <div class="chart-grid">
            {_score_histogram(scored_rows, "faithfulness", "Faithfulness distribution")}
            {_score_histogram(scored_rows, "relevance", "Relevance distribution")}
          </div>
          <div class="card table-card">
            <table>
              <thead><tr><th>ID</th><th>Query</th><th>Faithfulness</th><th>Relevance</th><th>Rationale</th></tr></thead>
              <tbody>{_generation_rows(generation_rows)}</tbody>
            </table>
          </div>
        </section>"""
    else:
        generation_section = """
        <section><h2>Generation quality</h2>
          <div class="card empty">Generation judging was skipped. Set
          <code>GROQ_API_KEY</code> and rerun <code>uv run python -m eval.run_eval</code>.
          </div>
        </section>"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LeaseGPT RAG Evaluation</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f4f1eb; --surface:#fffdf8; --text:#17211e;
      --muted:#68736e; --border:#d9d5ca; --blue:#315d58; --teal:#d7663e;
      --green:#287149; --red:#a53e35; --amber:#92671a; --ink:#173c38; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font:15px/1.55 "Avenir Next",Avenir,"Segoe UI",sans-serif;
      background:
        radial-gradient(circle at 8% 2%, rgba(215,102,62,.13), transparent 26rem),
        radial-gradient(circle at 94% 8%, rgba(49,93,88,.12), transparent 30rem),
        var(--bg); color:var(--text); }}
    main {{ max-width:1320px; margin:auto; padding:56px 32px 84px; }}
    h1,h2,h3 {{ font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif; }}
    h1 {{ margin:0; max-width:800px; font-size:clamp(2.6rem,6vw,5.25rem);
      line-height:.96; letter-spacing:-.055em; color:var(--ink); }}
    h2 {{ margin:52px 0 18px; font-size:1.75rem; letter-spacing:-.025em; }}
    h3 {{ margin:0 0 18px; font-size:1.15rem; }}
    .hero {{ position:relative; padding:26px 0 10px; }}
    .hero::after {{ content:""; position:absolute; right:2%; top:0; width:120px; height:120px;
      border:1px solid var(--border); border-radius:50%; box-shadow:
      -28px 24px 0 -1px var(--bg), -28px 24px 0 0 var(--teal); opacity:.8; z-index:-1; }}
    .eyebrow {{ display:inline-flex; align-items:center; gap:9px; margin:0 0 22px;
      color:var(--blue); font-size:.73rem; font-weight:700; letter-spacing:.16em;
      text-transform:uppercase; }}
    .eyebrow::before {{ content:""; width:26px; height:2px; background:var(--teal); }}
    .subtitle,.meta,.axis-note {{ color:var(--muted); }} .subtitle {{ font-size:1.05rem; }}
    .subtitle {{ max-width:570px; font-size:1.15rem; }}
    .meta {{ margin:20px 0 32px; font-size:.86rem; }} code {{ font-family:ui-monospace,monospace; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(185px,1fr)); gap:14px; }}
    .card {{ background:color-mix(in srgb,var(--surface) 94%,transparent);
      border:1px solid var(--border); border-radius:5px;
      box-shadow:0 14px 40px rgba(39,47,43,.055); }}
    .metric {{ position:relative; overflow:hidden; padding:21px 22px; }}
    .metric::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
      background:var(--teal); }} .metric:nth-child(even)::before {{ background:var(--blue); }}
    .metric span {{ color:var(--muted); display:block; font-size:.72rem; font-weight:700;
      letter-spacing:.09em; text-transform:uppercase; }}
    .metric strong {{ display:block; font-family:"Iowan Old Style",Georgia,serif;
      font-size:1.85rem; font-weight:600; margin-top:6px; }}
    .metric .delta {{ color:var(--green); font-weight:700; }}
    .chart-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
    .chart-card {{ padding:24px; min-height:390px; }}
    .chart {{ height:280px; display:flex; align-items:flex-end; gap:24px;
      border-bottom:1px solid var(--border); padding:28px 12px 0; }}
    .bar-group {{ flex:1; height:100%; display:flex; flex-direction:column; justify-content:flex-end;
      align-items:center; min-width:70px; }}
    .bars {{ width:100%; height:90%; display:flex; align-items:flex-end; justify-content:center; gap:8px; }}
    .bar-wrap {{ width:36%; height:100%; display:flex; flex-direction:column; justify-content:flex-end;
      align-items:center; color:var(--muted); font-size:.72rem; }}
    .bar {{ width:100%; min-height:2px; border-radius:2px 2px 0 0; background:var(--blue); }}
    .bar.constraint_rerank {{ background:var(--teal); }} .bar.score {{ background:var(--teal); }}
    .single {{ align-items:flex-end; }} .single .bar {{ width:55%; }}
    .bar-group strong {{ margin-top:8px; }} .legend {{ color:var(--muted); margin-top:16px; }}
    .legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px;
      background:var(--blue); margin:0 5px 0 14px; }}
    .legend i.constraint_rerank {{ background:var(--teal); }}
    .table-card {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; min-width:760px; }}
    th,td {{ text-align:left; padding:12px 14px; border-bottom:1px solid var(--border); }}
    th {{ color:var(--muted); font-size:.78rem; letter-spacing:.04em; text-transform:uppercase; }}
    tbody tr:last-child td {{ border-bottom:0; }}
    .prose {{ padding:22px 24px; }} .prose p:first-child {{ margin-top:0; }}
    .prose p:last-child {{ margin-bottom:0; }} .method-grid {{ display:grid;
      grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; }}
    .method-grid h3 {{ font-size:1rem; margin-bottom:8px; }}
    .method-grid p {{ color:var(--muted); margin:0; }}
    .toolbar {{ display:flex; justify-content:flex-end; margin:-48px 0 16px; }}
    select {{ font:inherit; padding:8px 12px; border:1px solid var(--border); border-radius:8px;
      background:var(--surface); color:var(--text); }}
    .pill {{ display:inline-block; border-radius:2px; padding:3px 8px; font-size:.7rem;
      font-weight:700; letter-spacing:.04em; text-transform:uppercase;
      background:#e2e8f0; color:#334155; white-space:nowrap; }}
    .pill.helped {{ background:#dcfce7; color:var(--green); }}
    .pill.hurt {{ background:#fee2e2; color:var(--red); }}
    .pill.unscored {{ background:#fef3c7; color:var(--amber); }}
    .empty {{ padding:22px; color:var(--muted); }}
    details {{ margin-top:20px; }} details pre {{ white-space:pre-wrap; padding:20px; overflow:auto; }}
    @media (max-width:800px) {{ main {{ padding:30px 16px 50px; }}
      .chart-grid {{ grid-template-columns:1fr; }} .toolbar {{ margin:0 0 12px; justify-content:flex-start; }}
      .chart {{ gap:8px; }} .hero::after {{ display:none; }} }}
    @media (prefers-color-scheme:dark) {{ :root {{ --bg:#111916; --surface:#18231f;
      --text:#e8ece7; --muted:#a4afa9; --border:#34413c; --ink:#edf4ef;
      --blue:#79aaa3; --teal:#e5835f; }}
      .pill {{ background:#334155; color:#e2e8f0; }} .pill.helped {{ background:#123c2c; }}
      .pill.hurt {{ background:#4a1d24; }} .pill.unscored {{ background:#473616; }} }}
  </style>
</head>
<body><main>
  <header class="hero">
    <p class="eyebrow">Retrieval intelligence report</p>
    <h1>LeaseGPT RAG evaluation</h1>
    <p class="subtitle">Retrieval quality and grounded-generation evidence</p>
    <p class="meta">Snapshot <code>{escape(metadata['snapshot_fetched_at'])}</code> ·
      digest <code>{escape(metadata['snapshot_digest'][:12])}</code> ·
      embedding <code>{escape(metadata['embedding_model'])}</code> ·
      generated <code>{escape(metadata['generated_at'])}</code></p>
    <div class="metrics">
      {_metric_card("Queries", str(metadata['query_count']))}
      {_metric_card("Snapshot", metadata['snapshot_fetched_at'])}
      {_metric_card("Embedding", metadata['embedding_model'].split('/')[-1])}
      {_metric_card("LLM judge", metadata['judge_status'].replace('_', ' '))}
    </div>
  </header>
  <section>
    <h2>What this evaluation tests</h2>
    <div class="card prose">
      <p>This evaluation asks two separate questions: <strong>does retrieval find the
      right listing records?</strong> and <strong>does the generated answer stay
      supported by the records it received?</strong> Keeping those stages separate
      makes failures diagnosable instead of collapsing quality into one opaque score.</p>
      <div class="method-grid">
        <div><h3>Dataset</h3><p>{metadata['query_count']} realistic Seattle rental
        searches are labeled against the frozen RentCast snapshot. Labels use stable
        listing IDs and observable fields such as bedrooms, price, neighborhood, and
        property type. Three unsupported or impossible requests are retained as
        qualitative hard cases and excluded from macro retrieval averages.</p></div>
        <div><h3>Recall@k</h3><p>Of the listings labeled relevant for a query, the
        fraction appearing in the first k retrieved results. Higher recall means fewer
        known-good listings were missed.</p></div>
        <div><h3>Precision@k</h3><p>Of the first k retrieved positions, the fraction
        occupied by labeled-relevant listings. Higher precision means less irrelevant
        context reaches the generator.</p></div>
        <div><h3>Macro averaging</h3><p>Recall and precision are calculated per labeled
        query and then averaged, so a query with many labels does not dominate the
        result. Scores are reported at k = 3, 5, and 10.</p></div>
      </div>
    </div>
  </section>
  <section>
    <h2>Retrieval comparison</h2>
    <div class="card prose">
      <p><strong>Baseline</strong> reproduces the app's dense FAISS similarity ranking
      with local <code>{escape(metadata['embedding_model'])}</code> embeddings.
      <strong>Constraint re-rank</strong> takes the first
      {metadata.get('rerank_pool_size', 25)} dense candidates, then prioritizes explicit
      bedroom, budget, neighborhood, and property-type matches. This is evaluation-only
      and does not change the LeaseGPT app.</p>
      <p>A chunk-size comparison is not meaningful for this snapshot: every synthesized
      listing is shorter than the app's split threshold and therefore already forms one
      document.</p>
    </div>
    <div class="metrics">
      {_metric_card("Baseline recall@5", _percent(baseline['recall@5']))}
      {_metric_card("Re-ranked recall@5", _percent(reranked['recall@5']), f"{retrieval_delta:+.1%}")}
      {_metric_card("Baseline precision@5", _percent(baseline['precision@5']))}
      {_metric_card("Re-ranked precision@5", _percent(reranked['precision@5']), f"{precision_delta:+.1%}")}
    </div>
    <div class="chart-grid">
      {_retrieval_chart(results, "recall", "Mean recall")}
      {_retrieval_chart(results, "precision", "Mean precision")}
    </div>
    <h2>Per-query retrieval</h2>
    <div class="toolbar"><label>Impact&nbsp;
      <select id="impact-filter"><option value="all">All queries</option>
        <option value="helped">Re-ranker helped</option><option value="hurt">Re-ranker hurt</option>
        <option value="same">No change</option><option value="unscored">Unscored hard cases</option>
      </select></label></div>
    <div class="card table-card"><table id="retrieval-table">
      <thead><tr><th>ID</th><th>Query</th><th>Labeling note</th><th>Gold</th><th>Baseline hits@5</th>
        <th>Re-ranked hits@5</th><th>Impact</th></tr></thead>
      <tbody>{_retrieval_rows(results)}</tbody>
    </table></div>
  </section>
  {generation_section.strip()}
  <section>
    <h2>How generation is judged</h2>
    <div class="card prose">
      <p>For each query, LeaseGPT generates an answer through the same conversational
      agent and RetrievalQA path used by the app. A separate deterministic-temperature
      Groq call then sees only the user query, the baseline top-4 retrieved listing
      texts, and the answer. It assigns independent 1–5 scores for
      <strong>faithfulness</strong> (every factual claim is supported) and
      <strong>relevance</strong> (the answer addresses the requested constraints), plus
      a short rationale.</p>
      <p>These scores are model judgments rather than human ground truth. The saved
      rubric and per-query rationales make them auditable, but repeated runs can still
      vary. Retrieval labels are curated examples rather than exhaustive relevance
      judgments, so precision should be interpreted comparatively.</p>
    </div>
  </section>
  <details><summary>Judge rubric</summary><pre class="card">{escape(results['rubric'])}</pre></details>
</main>
<script>
  const filter = document.getElementById("impact-filter");
  filter.addEventListener("change", () => {{
    document.querySelectorAll("#retrieval-table tbody tr").forEach(row => {{
      row.hidden = filter.value !== "all" && row.dataset.impact !== filter.value;
    }});
  }});
</script></body></html>
"""


def main() -> None:
    """Refresh only the HTML view from an existing JSON evaluation artifact."""
    eval_dir = Path(__file__).resolve().parent
    results_path = eval_dir / "results.json"
    output_path = eval_dir / "results.html"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    output_path.write_text(render_html(results), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
