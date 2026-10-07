"use strict";

const $ = (id) => document.getElementById(id);

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const escRe = (t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/* ---- highlight the words the user typed ---- */

const STOP = new Set("a an the and or of in for with to on at by from that this is are was were has have as it its".split(" "));

function termsOf(q) {
  const seen = new Set();
  return String(q || "").toLowerCase()
    .replace(/[^a-z0-9 ]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP.has(t) && !seen.has(t) && seen.add(t));
}

function hlAll(text, terms) {
  let out = esc(text);
  if (!terms.length) return out;
  const re = new RegExp("\\b(" + terms.map(escRe).join("|") + ")\\b", "gi");
  return out.replace(re, '<mark class="hl">$1</mark>');
}

function snippet(text, terms, len) {
  len = len || 240;
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

/* ---- shared bits ---- */

function badge(v) {
  return `<span class="badge ${esc(v)}">${esc(v)}</span>`;
}

function traceStep(line) {
  return `<div class="trace-step"><b>${esc(line.label)}</b> <span>size=${line.size}</span>${line.detail ? " — " + esc(line.detail) : ""}</div>`;
}

/* ============================================================
   CHAT VIEW — the RAG assistant
   ============================================================ */

const chat = $("chat");
let messageSeq = 0;

function addUser(text) {
  messageSeq++;
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="avatar">you</div><div class="body">
      <span class="label">you</span>${esc(text)}</div>`;
  chat.appendChild(el);
  return el;
}

function addTyping() {
  const el = document.createElement("div");
  el.className = "msg assistant typing-msg";
  el.innerHTML = `<div class="avatar">V</div><div class="body"><span class="typing">
    <i></i><i></i><i></i></span></div>`;
  chat.appendChild(el);
  scrollChat();
  return el;
}

function scrollChat() {
  chat.scrollTop = chat.scrollHeight;
}

function sentenceVerdictClass(v) {
  if (v && v.indexOf("SUPPORTED") === 0) return "dot-g";
  if (v === "WEAK") return "dot-a";
  return "dot-r";
}

function buildEvidence(d) {
  const qterms = termsOf(d.query);
  const chunks = d.retrieved.map((h, idx) => {
    const cited = d.verification.some((r, i) => (r.citations || [])[0] === h.rank);
    return `<div class="chunk" id="chunk-${messageSeq}-${idx}">
      <div class="chunk-head"><b>[${h.rank}]</b> <span class="t">${esc(h.title)}</span></div>
      <div class="chunk-meta">${esc(h.parent_doc)} · score ${h.score.toFixed(4)} · ${esc(h.chunk_id)}</div>
      <details><summary>${cited ? "cited by the answer — " : ""}read the passage</summary>
        <pre>${snippet(h.text, qterms)}</pre>
      </details>
    </div>`;
  }).join("");
  return `<div class="fold"><summary>evidence — ${d.retrieved.length} passages retrieved</summary>
    <div class="fold-body">${chunks}</div></div>`;
}

function buildTraceFold(d) {
  if (!d.trace) return "";
  const terms = (d.trace.query_terms || []).join(", ");
  const elim = d.trace.eliminated && d.trace.eliminated.length ? " · eliminated: " + esc(d.trace.eliminated.join(", ")) : "";
  return `<div class="fold"><summary>query trace — how this was retrieved</summary>
    <div class="fold-body"><div class="trace-step"><b>terms</b> <span>${terms}</span>${elim}</div>
    ${(d.trace.term_stats ? Object.keys(d.trace.term_stats) : []).length
      ? Object.entries(d.trace.term_stats).slice(0, 6).map(([t, st]) =>
          `<div class="trace-step"><b>${esc(t)}</b> <span>df=${st.df} · idf=${st.idf.toFixed(3)}</span></div>`).join("")
      : ""}</div></div>`;
}

function addAssistant(d) {
  messageSeq++;
  const qterms = termsOf(d.query);

  const sentences = d.answer.length
    ? d.answer.map((s, i) => {
        const row = d.verification[i];
        const cites = (s.citations || []).map((c) => `<sup class="cite" data-seqs="${messageSeq}" data-rank="${c}">${c}</sup>`).join("");
        const dots = `<span class="dots"><i class="${sentenceVerdictClass(row ? row.verdict : "SUPPORTED")}" title=""></i></span>`;
        return `<p>${esc(s.text)} ${cites} ${dots}</p>`;
      }).join("")
    : "<p>(no answer sentences could be verified — nothing was safe to show.)</p>";

  const counts = Object.entries(d.verdicts).map(([v, n]) => badge(v) + " x" + n).join(" ");
  const strip = `<div class="verdict-strip">${counts}
    <span>${d.answer.length ? "sentence dots: " + d.answer.map((s, i) => {
      const v = d.verification[i] ? d.verification[i].verdict : "SUPPORTED";
      return `<i class="dots"><i class="${sentenceVerdictClass(v)}"></i></i>`;
    }).join(" ") : ""}</span>
    <span>· ${d.elapsed_ms} ms</span></div>`;

  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = `<div class="avatar">V</div><div class="body">
      <div class="answer">${sentences}</div>
      ${strip}
      ${buildEvidence(d)}
      ${buildTraceFold(d)}
    </div>`;
  chat.appendChild(el);

  el.querySelectorAll("sup.cite").forEach((sup) =>
    sup.addEventListener("click", () => {
      const fold = el.querySelector(".fold");
      fold.open = true;
      const target = el.querySelector(`#chunk-${messageSeq}-${Number(sup.dataset.rank) - 1}`);
      if (target) {
        (target.querySelector("details") || target).open = true;
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        target.classList.add("hot");
        setTimeout(() => target.classList.remove("hot"), 2000);
      }
    })
  );

  scrollChat();
  return el;
}

async function askQuestion(text) {
  const explain = $("opt-explain").checked;
  const k = parseInt($("opt-k").value, 10) || 5;
  const sentences = parseInt($("opt-sentences").value, 10) || 3;
  addUser(text);
  const typing = addTyping();
  const go = $("go");
  go.disabled = true;
  $("suggestions").classList.add("hidden");
  const status = $("status");
  status.classList.add("hidden");
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text, k, sentences, explain, static: 0.0 }),
    });
    const data = await res.json();
    typing.remove();
    if (!res.ok) throw new Error(data.error || res.statusText);
    addAssistant(data);
  } catch (err) {
    typing.remove();
    status.textContent = "error: " + err.message;
    status.classList.remove("hidden");
  } finally {
    go.disabled = false;
  }
}

$("suggestions").querySelectorAll("[data-ask]").forEach((chip) =>
  chip.addEventListener("click", () => askQuestion(chip.dataset.ask))
);

/* ============================================================
   ENGINE INSPECT — technical renderers (hidden from chat)
   ============================================================ */

const ENGINE_TABS = ["search", "bool", "term", "verify"];
const RENDERERS = {};

function renderSearch(d) {
  const qt = termsOf(d.query);
  const trace = d.trace && d.trace.query_terms.length
    ? `<div class="trace-step"><b>query analysis</b> <span>terms: ${d.trace.query_terms.join(", ")} · candidates ${d.trace.candidates}${d.trace.eliminated.length ? " · eliminated: " + esc(d.trace.eliminated.join(", ")) : ""}</span></div>`
    : "";
  const rows = d.hits.map((h) => `<div class="hit">
      <div class="head"><span class="score">${h.rank}. ${h.score.toFixed(4)}</span> [${esc(h.chunk_id)}] doc=${esc(h.parent_doc)}
        · <span class="title">${esc(h.title)}</span></div>
      <pre>${snippet(h.text, qt)}</pre>
      <div class="component-bar">
        <span>base ${h.base.toFixed(3)}</span><div class="pct cmp-base" style="width:${((h.base) * 60).toFixed(1)}%"></div>
        <span>prox ${h.proximity.toFixed(3)}</span><div class="pct cmp-prox" style="width:${(h.proximity * 60).toFixed(1)}%"></div>
        <span>g ${h.g.toFixed(3)}</span><div class="pct cmp-static" style="width:${(h.g * 60).toFixed(1)}%"></div>
      </div>
    </div>`).join("");
  return `<div class="card"><h3>Top-${d.hits.length} hits · <span class="hit-note">yellow = your words</span></h3>${trace}${rows || "<p>(no hits)</p>"}</div>`;
}

function renderBool(d) {
  const hits = d.n_results
    ? d.hits.map((h) => `<div class="hit"><div class="head">[${esc(h.chunk_id)}] doc=${esc(h.parent_doc)} · <span class="title">${esc(h.title)}</span></div></div>`).join("")
    : "<p>(no matches)</p>";
  return `<div class="card"><h3>${d.n_results} matching docs</h3>${d.trace.map(traceStep).join("")}${hits}</div>`;
}

function renderTerm(d) {
  const rows = d.postings_sample.map((p) =>
    `<tr><td>${esc(p.chunk_id)}</td><td class="num">${p.tf}</td><td class="num">${p.tf_title}</td><td>${p.positions.join(", ")}</td></tr>`).join("");
  return `<div class="card">
    <h3>${esc(d.term)} — df=${d.df} · idf=${d.idf} · postings=${d.postings_len}</h3>
    <table class="term-table"><tr><th>chunk</th><th>tf</th><th>in title</th><th>positions</th></tr>${rows}</table>
    <h4>skip-list pointers</h4>
    <div>${d.skips.map(([i, doc]) => `<span class="pill">${i} → ${esc(doc)}</span>`).join(" ") || "— (list < 32)"}</div>
    ${d.champions.length ? `<h4>champion list</h4><div>${d.champions.map((c) => `<span class="pill">${esc(c)}</span>`).join(" ")}</div>` : ""}
  </div>`;
}

function renderVerify(d) {
  const qt = termsOf(d.claim);
  const rows = d.rows.map((r) => `<tr>
      <td class="num">${r.rank}</td>
      <td>[${esc(r.chunk_id)}]</td>
      <td>${esc(r.title)}<br><span class="snippet">${snippet(r.text, qt, 130)}</span></td>
      <td class="num">${r.cosine.toFixed(4)}</td>
      <td class="num">${r.novelty_penalty.toFixed(3)}</td>
      <td class="num">${r.support.toFixed(4)}</td>
      <td>${r.novel_terms.length ? esc(r.novel_terms.join(" ")) : "-"}</td>
    </tr>`).join("");
  return `<div class="card">
    <div class="verdict-strip"><span class="badge mono">terms ${esc(d.terms.join(" "))}</span>
      <strong>verdict:</strong> ${badge(d.verdict)} <span class="badge mono">best ${d.best_support.toFixed(4)}</span></div>
    <table><tr><th>#</th><th>chunk</th><th>title</th><th class="num">cos</th><th class="num">novelty</th><th class="num">support</th><th>absent terms</th></tr>${rows}</table>
  </div>`;
}

RENDERERS.search = renderSearch;
RENDERERS.bool = renderBool;
RENDERERS.term = renderTerm;
RENDERERS.verify = renderVerify;

/* ---- engine view switch ---- */

let activeTab = "search";
let view = "chat";

function switchView(next) {
  view = next;
  const engine = next === "engine";
  $("engine-view").classList.toggle("active", engine);
  $("chat-view").classList.toggle("engine-off", engine);
  $("engine-toggle").textContent = engine ? "Back to chat" : "Inspect engine";
  $("engine-toggle").classList.toggle("active", engine);
  $("input").placeholder = engine ? { search: "search the corpus…", bool: 'boolean query, e.g. "antibiotic resistance" AND bacteria', term: "one term, e.g. resist", verify: "paste a claim to verify…" }[activeTab] : "Ask your scientific question…";
  if (!engine) $("input").focus();
}

$("engine-toggle").addEventListener("click", () => switchView(view === "engine" ? "chat" : "engine"));

document.querySelectorAll(".tab").forEach((b) =>
  b.addEventListener("click", () => {
    activeTab = b.dataset.tab;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + activeTab));
    $("engine-results").innerHTML = "";
    $("input").placeholder = { search: "search the corpus…", bool: 'boolean query, e.g. "antibiotic resistance" AND bacteria', term: "one term, e.g. resist", verify: "paste a claim to verify…" }[activeTab];
  })
);

document.querySelectorAll("[data-engine]").forEach((chip) =>
  chip.addEventListener("click", () => {
    $("input").value = chip.dataset.value;
    runEngine();
  })
);

async function runEngine() {
  const input = $("input").value.trim();
  if (!input) return;
  const k = parseInt($("opt-k").value, 10) || 5;
  const explain = $("opt-explain").checked;
  const payload = { query: input, k, explain };
  if (activeTab === "verify") payload.claim = input;
  if (activeTab === "term") payload.term = input.split(/\s+/)[0];
  const endpoint = { search: "/api/search", bool: "/api/bool", term: "/api/term", verify: "/api/verify" }[activeTab];
  const status = $("status");
  const go = $("go");
  go.disabled = true;
  status.classList.add("hidden");
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    $("engine-results").innerHTML = RENDERERS[activeTab](data);
  } catch (err) {
    status.textContent = "error: " + err.message;
    status.classList.remove("hidden");
  } finally {
    go.disabled = false;
  }
}

/* ---- composer wiring ---- */

function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 140) + "px";
}

function submit() {
  const text = $("input").value.trim();
  if (!text) return;
  if (view === "engine") {
    runEngine();
    return;
  }
  $("input").value = "";
  autoGrow($("input"));
  askQuestion(text);
}

$("go").addEventListener("click", submit);
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submit();
  }
});
$("input").addEventListener("input", () => autoGrow($("input")));

/* ---- boot ---- */

(async function init() {
  const hero = `<div class="msg assistant"><div class="avatar">V</div><div class="body">
    <p><strong>Ask a biomedical question.</strong> VeriRAG reads 5,183 papers from BEIR SciFact,
    generates an answer sentence by sentence, and verify each sentence against the passage it cites
    before showing it — supported citations get a green dot, doubted ones amber or red.</p>
    <p>Type a question below, or pick an example. The technical machinery — inverted index, Boolean
    queries, retrieval traces — stays under <em>Inspect engine</em>.</p></div></div>`;
  chat.appendChild(heroToEl(hero));
  scrollChat();
  try {
    const res = await fetch("/api/stats", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const s = await res.json();
    $("stat-chunks").textContent = s.n_chunks;
    $("stat-terms").textContent = s.n_terms;
    $("stat-docs").textContent = s.n_docs;
  } catch (e) { /* optional */ }
})();

function heroToEl(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstChild;
}