"use strict";

const TABS = ["ask", "search", "bool", "term", "verify"];
const LABEL = { ask: "Ask", search: "Search", bool: "Boolean", term: "Term", verify: "Verify" };

const $ = (id) => document.getElementById(id);
const states = { ask: false, search: true, bool: false, term: false, verify: false };

let activeTab = "ask";

/* ---- tab switching ---- */

function setTab(tab) {
  activeTab = tab;
  for (const t of TABS) {
    $("panel-" + t).classList.toggle("active", t === tab);
    document.querySelector(`[data-tab="${t}"]`).classList.toggle("active", t === tab);
  }
  const placeholder = {
    ask: "ask a question…",
    search: "query (e.g. soil bacteria genes)…",
    bool: 'boolean expression, e.g. "antibiotic resistance" AND bacteria …',
    term: "one term, e.g. resist …",
    verify: "paste a claim to check…",
  }[tab];
  $("input").placeholder = placeholder;
  $("input").focus();
}

document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => setTab(btn.dataset.tab))
);

/* ---- example chips ---- */

document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    setTab(chip.dataset.target);
    $("input").value = chip.dataset.value;
    run();
  })
);

/* ---- rendering helpers ---- */

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---- highlight the words the user typed ---- */

const STOP = new Set("a an the and or of in for with to on at by from that this is are was were has have as it its".split(" "));

function termsOf(q) {
  const seen = new Set();
  return String(q || "").toLowerCase()
    .replace(/[^a-z0-9 ]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP.has(t) && !seen.has(t) && seen.add(t));
}

const escRe = (t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function hlAll(text, terms) {
  let out = esc(text);
  if (!terms.length) return out;
  const re = new RegExp("\\b(" + terms.map(escRe).join("|") + ")\\b", "gi");
  return out.replace(re, '<mark class="hl">$1</mark>');
}

function snippet(text, terms, len) {
  len = len || 260;
  if (text.length <= len) return hlAll(text, terms);
  let pos = -1;
  const lower = text.toLowerCase();
  for (const t of terms) {
    const i = lower.indexOf(t);
    if (i !== -1) { pos = i; break; }
  }
  if (pos < 0) return hlAll(text.slice(0, len), terms) + "…";
  const start = Math.max(0, pos - 60);
  const head = start > 0 ? "…" : "";
  return head + hlAll(text.slice(start, start + len), terms) + "…";
}

function badge(v) {
  return `<span class="badge ${esc(v)}">${esc(v)}</span>`;
}

function renderTrace(steps) {
  if (!steps || !steps.length) return "";
  return (
    "<h4>query-processor trace</h4>" +
    steps.map((s) => `<div class="trace-step"><b>${esc(s.label)}</b> <span>size=${s.size}</span>${s.detail ? " — " + esc(s.detail) : ""}</div>`).join("")
  );
}

function renderHits(rows, showText, terms) {
  return rows.map((h) => {
    const meta = `${h.rank}. ${h.score.toFixed(4)} [${esc(h.chunk_id)}] doc=${esc(h.parent_doc)}`;
    return `<div class="hit">
      <div class="head"><span class="score">${meta}</span> · <span class="title">${esc(h.title)}</span></div>
      ${showText ? `<pre>${snippet(h.text, terms || [])}</pre>` : ""}
    </div>`;
  }).join("");
}

function componentBar(r) {
  const total = Math.max(r.base + r.proximity + r.g, 1e-9);
  const pct = (x) => ((x / total) * 100).toFixed(1);
  return `<div class="component-bar">
    <span title="cosine base">base ${r.base.toFixed(3)}</span>
    <div class="pct cmp-base" style="width:${pct(r.base)}%"></div>
    <span title="proximity">prox ${r.proximity.toFixed(3)}</span>
    <div class="pct cmp-prox" style="width:${pct(r.proximity)}%"></div>
    <span title="static quality">g ${r.g.toFixed(3)}</span>
    <div class="pct cmp-static" style="width:${pct(r.g)}%"></div>
  </div>`;
}

/* ---- per-tab renderers ---- */

function renderAsk(d) {
  const qterms = termsOf(d.query);
  const counts = Object.entries(d.verdicts).map(([v, n]) => badge(v) + " x" + n).join(" ");
  const trace = d.trace
    ? `<div class="trace-step"><b>query analysis</b> <span>terms: ${d.trace.query_terms.join(", ")}</span>${d.trace.eliminated.length ? " — eliminated low-idf: " + esc(d.trace.eliminated.join(", ")) : ""}</div>`
    : "";
  const answers = d.answer.length
    ? d.answer.map((s) => {
        const cites = s.citations.map((c) => `[${c}]`).join("");
        return `<div class="answer-item"><div class="text">${esc(s.text)} ${cites}</div>
          <div class="meta">relevance ${s.relevance.toFixed(3)} · sources ${esc(s.sources.join(", "))}</div></div>`;
      }).join("")
    : '<div class="answer-item">(no answer sentences produced)</div>';

  const verif = d.verification.length
    ? `<table>
        <tr><th>#</th><th>verdict</th><th>support</th><th>cos</th><th>novel</th><th>cites</th><th>claim</th><th>match</th></tr>
        ${d.verification.map((r, i) => `<tr>
          <td>${i + 1}</td>
          <td>${badge(r.verdict)}</td>
          <td class="num">${r.support.toFixed(4)}</td>
          <td class="num">${r.cosine.toFixed(4)}</td>
          <td class="num">${r.novelty_penalty.toFixed(3)}</td>
          <td class="num">${esc(r.citations.join(","))}</td>
          <td>${esc(r.claim.slice(0, 90))}</td>
          <td>${r.cite_mismatch ? '<span class="badge mono">better in rank ' + r.best_rank + "</span>"
            : r.novel_terms.length ? '<span class="badge mono">novel: ' + esc(r.novel_terms.slice(0, 4).join(" ")) + "</span>" : "-"}</td>
        </tr>`).join("")}
      </table>`
    : "";

  return `<div class="card">
    <div class="verdict-strip"><span class="badge mono">${LABEL.ask} · ${d.elapsed_ms} ms</span> ${counts}</div>
    ${trace}
    <h4>generated answer (extractive, verified)</h4>
    ${answers}
    ${verif}
    <button class="chip show-retrieved">show ${d.retrieved.length} retrieved chunks</button>
    <div class="hidden retrieved-box"><h4>retrieved evidence chunks</h4>${renderHits(d.retrieved, true, qterms)}</div>
  </div>`;
}

function renderSearch(d) {
  const qterms = termsOf(d.query);
  const trace = d.trace && d.trace.query_terms.length
    ? `<div class="trace-step"><b>query analysis</b> <span>terms: ${d.trace.query_terms.join(", ")} · candidates ${d.trace.candidates}${d.trace.eliminated.length ? " · eliminated: " + esc(d.trace.eliminated.join(", ")) : ""}</span></div>`
    : "";
  const rows = d.hits.map((h) => `<div class="hit">
      <div class="head"><span class="score">${h.rank}. ${h.score.toFixed(4)}</span> [${esc(h.chunk_id)}] doc=${esc(h.parent_doc)}
        · <span class="title">${esc(h.title)}</span></div>
      <pre>${snippet(h.text, qterms)}</pre>
      ${componentBar(h)}
    </div>`).join("");
  return `<div class="card"><h3>Top-${d.hits.length} hits · <span class="hit-note">yellow = your words</span></h3>${trace}${rows || "<p>(no hits)</p>"}</div>`;
}

function renderBool(d) {
  const steps = d.trace.map((s) => `<div class="trace-step"><b>${esc(s.label)}</b> <span>size=${s.size}</span>${s.detail ? " — " + esc(s.detail) : ""}</div>`).join("");
  const hits = d.n_results
    ? d.hits.map((h) => `<div class="hit"><div class="head">[${esc(h.chunk_id)}] doc=${esc(h.parent_doc)} · <span class="title">${esc(h.title)}</span></div></div>`).join("")
    : "<p>(no matches)</p>";
  return `<div class="card"><h3>${d.n_results} matching docs</h3>${renderTrace(d.trace)}${hits}</div>`;
}

function renderTerm(d) {
  const rows = d.postings_sample.map((p) =>
    `<tr><td>${esc(p.chunk_id)}</td><td class="num">${p.tf}</td><td class="num">${p.tf_title}</td><td>${p.positions.join(", ")}</td></tr>`).join("");
  return `<div class="card">
    <h3>${esc(d.term)} — df=${d.df} · idf=${d.idf} · postings=${d.postings_len}</h3>
    <table class="term-table"><tr><th>chunk</th><th>tf</th><th>in title</th><th>positions</th></tr>${rows}</table>
    <h4>skip-list pointers ${d.postings_len >= 32 ? "(step ~√df)" : "(list < 32, no skips)"}</h4>
    <div>${d.skips.map(([i, doc]) => `<span class="pill">${i} → ${esc(doc)}</span>`).join(" ")}</div>
    ${d.champions.length ? `<h4>champion list (top docs)</h4><div>${d.champions.map((c) => `<span class="pill">${esc(c)}</span>`).join(" ")}</div>` : ""}
  </div>`;
}

function renderVerify(d) {
  const qt = termsOf(d.claim);
  const rows = d.rows.map((r) => `<tr>
      <td class="num">${r.rank}</td>
      <td>[${esc(r.chunk_id)}]</td>
      <td>${esc(r.title)}<br><span class="snippet">${snippet(r.text, qt, 140)}</span></td>
      <td class="num">${r.cosine.toFixed(4)}</td>
      <td class="num">${r.novelty_penalty.toFixed(3)}</td>
      <td class="num">${r.support.toFixed(4)}</td>
      <td>${r.novel_terms.length ? esc(r.novel_terms.join(" ")) : "-"}</td>
    </tr>`).join("");
  return `<div class="card">
    <div class="verdict-strip"><span class="badge mono">terms ${esc(d.terms.join(" "))}</span></div>
    <div class="verdict-strip"><strong>verdict:</strong> ${badge(d.verdict)} <span class="badge mono">best support ${d.best_support.toFixed(4)} (high ${d.high} / low ${d.low})</span></div>
    <table><tr><th>#</th><th>chunk</th><th>title</th><th class="num">cos</th><th class="num">novelty</th><th class="num">support</th><th>absent terms</th></tr>${rows}</table>
  </div>`;
}

const RENDERERS = { ask: renderAsk, search: renderSearch, bool: renderBool, term: renderTerm, verify: renderVerify };

/* ---- run ---- */

async function run() {
  const input = $("input").value.trim();
  if (!input) return;
  const k = parseInt($("opt-k").value, 10) || 5;
  const sentences = parseInt($("opt-sentences").value, 10) || 3;
  const explain = $("opt-explain").checked;

  const payload = { query: input, k, sentences, explain, static: 0.0 };
  if (activeTab === "verify") payload.claim = input;
  if (activeTab === "term") payload.term = input.split(/\s+/)[0];
  if (activeTab === "bool") payload.query = input;

  const endpoint = { ask: "/api/ask", search: "/api/search", bool: "/api/bool", term: "/api/term", verify: "/api/verify" }[activeTab];

  const status = $("status");
  status.classList.remove("ok", "hidden");
  status.textContent = "running " + LABEL[activeTab] + "…";
  const go = $("go");
  go.disabled = true;

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    $("results").innerHTML =
      `<div class="card"><div class="verdict-strip"><span class="badge mono">${esc(LABEL[activeTab])} · ${esc(input)}</span></div>${RENDERERS[activeTab](data)}</div>`;
    status.classList.add("hidden");

    $("results").querySelectorAll(".show-retrieved").forEach((btn) =>
      btn.addEventListener("click", () => {
        const box = btn.parentElement.querySelector(".retrieved-box");
        box.classList.toggle("hidden");
        btn.textContent = box.classList.contains("hidden")
          ? `show ${box.querySelectorAll(".hit").length} retrieved chunks`
          : "hide retrieved chunks";
      })
    );
  } catch (err) {
    status.textContent = "error: " + err.message;
  } finally {
    go.disabled = false;
  }
}

$("go").addEventListener("click", run);
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") run();
});

/* ---- bootstrap ---- */

(async function init() {
  try {
    const res = await fetch("/api/stats", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const s = await res.json();
    $("stat-chunks").textContent = s.n_chunks;
    $("stat-terms").textContent = s.n_terms;
    $("stat-docs").textContent = s.n_docs;
  } catch (e) { /* server-side stats unavailable */ }
  setTab("ask");
})();