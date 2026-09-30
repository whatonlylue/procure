import { useCallback, useEffect, useRef, useState } from "react";
import {
  Doc,
  Health,
  Hit,
  Job,
  McpLogLine,
  McpStatus,
  SkillStatus,
  TagCount,
  WatchFolder,
  addUrl,
  addWatch,
  cancelJob,
  deleteDoc,
  getJob,
  health as fetchHealth,
  installSkills,
  listDocs,
  listTags,
  listWatches,
  mcpLogs,
  mcpStart,
  mcpStatus,
  mcpStop,
  reingestDoc,
  removeWatch,
  search,
  setDocTags,
  skillsStatus,
  syncWatches,
  updateSettings,
  uploadFilesAsync,
} from "./api";
import { THEMES, ThemeName, applyTheme, getInitialTheme } from "./theme";

type Tab = "search" | "library" | "sources" | "mcp";

function fmt(n: number | undefined): string {
  return typeof n === "number" && Number.isFinite(n) ? `${Math.round(n * 100)}%` : "—";
}

function parseTags(s: string): string[] {
  return s.split(",").map((t) => t.trim()).filter(Boolean);
}

const GREETINGS = [
  "What do you seek?",
  "Seek, and you shall find.",
  "Your library remembers everything.",
  "What are you hunting today?",
  "Your second mind awaits.",
  "What mystery shall we solve?",
  "Ask, and it shall be found.",
  "What did you stash away?",
];

function pickGreeting(): string {
  // One saying per session, advancing to the next on each fresh launch.
  try {
    const pinned = sessionStorage.getItem("procure-greeting");
    if (pinned !== null) {
      const i = Number(pinned);
      if (Number.isInteger(i) && i >= 0 && i < GREETINGS.length) return GREETINGS[i];
    }
    const last = Number(localStorage.getItem("procure-greeting-last") ?? "NaN");
    const next = Number.isInteger(last) && last >= 0 ? (last + 1) % GREETINGS.length : 0;
    localStorage.setItem("procure-greeting-last", String(next));
    sessionStorage.setItem("procure-greeting", String(next));
    return GREETINGS[next];
  } catch {
    return GREETINGS[0];
  }
}

export default function App() {
  const [tab, setTab] = useState<Tab>("search");
  const [theme, setTheme] = useState<ThemeName>(getInitialTheme);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [tags, setTags] = useState<TagCount[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<Hit[]>([]);
  const [searched, setSearched] = useState("");
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [drag, setDrag] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [topK, setTopK] = useState(10);
  const [backend, setBackend] = useState<Health | null>(null);
  const [mcp, setMcp] = useState<McpStatus | null>(null);
  const [mcpLines, setMcpLines] = useState<McpLogLine[]>([]);
  const [copied, setCopied] = useState(false);
  const [skill, setSkill] = useState<SkillStatus | null>(null);
  // Search scope.
  const [filterTags, setFilterTags] = useState<Set<string>>(new Set());
  const [scopeDocs, setScopeDocs] = useState<Set<string>>(new Set());
  const [scopeSource, setScopeSource] = useState("");
  const [scopeDocType, setScopeDocType] = useState("");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [greeting] = useState<string>(pickGreeting);
  // Library tag editor.
  const [editingTags, setEditingTags] = useState<string | null>(null);
  const [tagDraft, setTagDraft] = useState("");
  // Sources tab.
  const [url, setUrl] = useState("");
  const [urlTags, setUrlTags] = useState("");
  const [watches, setWatches] = useState<WatchFolder[]>([]);
  const [lastScan, setLastScan] = useState<string | null>(null);
  const [watchPath, setWatchPath] = useState("");
  const [watchRecursive, setWatchRecursive] = useState(true);
  const fileRef = useRef<HTMLInputElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const mcpNext = useRef(0);
  const mcpLogRef = useRef<HTMLPreElement>(null);
  const mcpFollow = useRef(true);
  const pollRef = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [d, h, m, t, w] = await Promise.all([
        listDocs(),
        fetchHealth(),
        mcpStatus(),
        listTags(),
        listWatches(),
      ]);
      setDocs(d);
      setBackend(h);
      setMcp(m);
      setTags(t);
      setWatches(w.folders);
      setLastScan(w.last_scan);
    } catch (e) {
      setNotice(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    return () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
    };
  }, []);

  function stopPoll() {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  // While the MCP tab is open, poll status + append new server output.
  useEffect(() => {
    if (tab !== "mcp") return;
    let alive = true;
    async function poll() {
      try {
        const [s, l] = await Promise.all([mcpStatus(), mcpLogs(mcpNext.current)]);
        if (!alive) return;
        setMcp(s);
        mcpNext.current = l.next;
        if (l.lines.length) {
          setMcpLines((prev) => [...prev, ...l.lines].slice(-1000));
        }
      } catch (e) {
        if (alive) setNotice(String(e));
      }
    }
    poll();
    const id = setInterval(poll, 2000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [tab]);

  // Follow the log tail unless the user scrolled up.
  useEffect(() => {
    const el = mcpLogRef.current;
    if (el && mcpFollow.current) el.scrollTop = el.scrollHeight;
  }, [mcpLines]);

  // Skill install state, refreshed whenever the MCP tab opens.
  useEffect(() => {
    if (tab !== "mcp") return;
    let alive = true;
    skillsStatus()
      .then((s) => { if (alive) setSkill(s); })
      .catch((e) => { if (alive) setNotice(String(e)); });
    return () => { alive = false; };
  }, [tab]);

  // Keyboard: "/" or Cmd/Ctrl+K focuses search; 1/2 switch tabs.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null;
      const typing = !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA");
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setTab("search");
        searchRef.current?.focus();
        searchRef.current?.select();
      } else if (e.key === "/" && !typing) {
        e.preventDefault();
        setTab("search");
        searchRef.current?.focus();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function summarizeResults(res: unknown): string {
    if (!Array.isArray(res)) return "Ingest complete.";
    const docs = res as { status?: string; filename?: string }[];
    const ready = docs.filter((d) => d.status === "ready").length;
    const dup = docs.filter((d) => d.status === "duplicate").length;
    const failed = docs.filter((d) => d.status === "failed");
    const cancelled = docs.filter((d) => d.status === "cancelled").length;
    const parts = [`Ingested ${ready}`];
    if (dup) parts.push(`${dup} duplicate${dup === 1 ? "" : "s"} skipped`);
    if (cancelled) parts.push(`${cancelled} cancelled`);
    if (failed.length) parts.push(`Failed: ${failed.map((f) => f.filename).join(", ")}`);
    return parts.join(". ") + ".";
  }

  function pollJob(id: string) {
    stopPoll();
    pollRef.current = window.setInterval(async () => {
      try {
        const j = await getJob(id);
        setJob(j);
        if (j.status === "done" || j.status === "failed" || j.status === "cancelled") {
          stopPoll();
          setBusy("");
          setNotice(j.status === "failed" ? `Job failed: ${j.error}` : summarizeResults(j.result));
          await refresh();
        }
      } catch (e) {
        stopPoll();
        setBusy("");
        setNotice(String(e));
      }
    }, 500);
  }

  async function handleFiles(files: FileList | File[]) {
    const arr = Array.from(files);
    if (!arr.length) return;
    setNotice("");
    setBusy(`Uploading ${arr.length} file(s)…`);
    try {
      const j = await uploadFilesAsync(arr);
      setJob(j);
      pollJob(j.job_id);
    } catch (e) {
      setBusy("");
      setNotice(String(e));
    }
  }

  async function cancelCurrentJob() {
    if (!job) return;
    try {
      const j = await cancelJob(job.job_id);
      setJob(j);
    } catch (e) {
      setNotice(String(e));
    }
  }

  async function runSearch(e?: React.FormEvent) {
    e?.preventDefault();
    if (!query.trim() || busy) return;
    setBusy("Searching…");
    setNotice("");
    try {
      const res = await search(query, topK, {
        docIds: scopeDocs.size ? [...scopeDocs] : undefined,
        tags: filterTags.size ? [...filterTags] : undefined,
        source: scopeSource || undefined,
        docType: scopeDocType || undefined,
      });
      setHits(res);
      setSearched(query);
    } catch (err) {
      setNotice(String(err));
    } finally {
      setBusy("");
    }
  }

  function clearSearch() {
    setQuery("");
    setHits([]);
    setSearched("");
  }

  function toggle(id: string) {
    setSelected((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  }

  function toggleTag(t: string) {
    setFilterTags((s) => {
      const n = new Set(s);
      if (n.has(t)) n.delete(t);
      else n.add(t);
      return n;
    });
  }

  function toggleScopeDoc(id: string) {
    setScopeDocs((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  }

  async function doDelete(id: string) {
    setBusy("Deleting…");
    try {
      await deleteDoc(id);
      setSelected((s) => {
        const n = new Set(s);
        n.delete(id);
        return n;
      });
      await refresh();
    } catch (e) {
      setNotice(String(e));
    } finally {
      setBusy("");
    }
  }

  async function doReingest(id: string) {
    setBusy("Re-ingesting…");
    try {
      await reingestDoc(id);
      await refresh();
      setNotice("Re-ingest complete.");
    } catch (e) {
      setNotice(String(e));
    } finally {
      setBusy("");
    }
  }

  async function saveTags(id: string) {
    try {
      await setDocTags(id, parseTags(tagDraft));
      setEditingTags(null);
      await refresh();
    } catch (e) {
      setNotice(String(e));
    }
  }

  async function toggleMcp() {
    const running = mcp?.running ?? false;
    setBusy(running ? "Stopping MCP server…" : "Starting MCP server…");
    setNotice("");
    try {
      setMcp(running ? await mcpStop() : await mcpStart());
    } catch (e) {
      setNotice(String(e));
    } finally {
      setBusy("");
    }
  }

  async function copyMcpUrl() {
    if (!mcp) return;
    try {
      await navigator.clipboard.writeText(mcp.url);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = mcp.url;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  async function doInstallSkill(force: boolean) {
    if (busy) return;
    setBusy(force ? "Reinstalling skill…" : "Installing skill…");
    setNotice("");
    try {
      const results = await installSkills(force);
      const errs = results.filter((r) => r.action === "error");
      setNotice(
        errs.length
          ? `Skill install errors: ${errs.map((r) => `${r.id}: ${r.detail}`).join("; ")}`
          : `Skill ${results.map((r) => `${r.id} ${r.action}`).join(", ")}.`
      );
      setSkill(await skillsStatus());
    } catch (e) {
      setNotice(String(e));
    } finally {
      setBusy("");
    }
  }

  function fmtUptime(s: number): string {
    if (s < 60) return `${Math.floor(s)}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  }

  async function changeSetting(patch: { search?: string; rerank?: string }) {
    setNotice("");
    try {
      const s = await updateSettings(patch);
      setBackend((b) => (b ? { ...b, search: s.search, rerank: s.rerank } : b));
    } catch (e) {
      setNotice(String(e));
    }
  }

  async function handleAddUrl(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim() || busy) return;
    setBusy("Fetching URL…");
    setNotice("");
    try {
      const doc = await addUrl(url.trim(), parseTags(urlTags));
      setUrl("");
      setUrlTags("");
      setNotice(
        doc.status === "duplicate"
          ? `Already in library as ${doc.filename}.`
          : doc.status === "failed"
            ? `Failed: ${doc.error}`
            : `Added ${doc.filename} (${doc.chunk_count} chunks).`
      );
      await refresh();
    } catch (e) {
      setNotice(String(e));
    } finally {
      setBusy("");
    }
  }

  async function handleAddWatch(e: React.FormEvent) {
    e.preventDefault();
    if (!watchPath.trim() || busy) return;
    setBusy("Adding watch…");
    setNotice("");
    try {
      await addWatch(watchPath.trim(), watchRecursive);
      setWatchPath("");
      setNotice("Watching folder — first sync running in background.");
      const j = await syncWatches();
      setJob(j);
      pollJob(j.job_id);
      await refresh();
    } catch (e) {
      setNotice(String(e));
    } finally {
      setBusy("");
    }
  }

  async function handleRemoveWatch(id: number) {
    setNotice("");
    try {
      await removeWatch(id);
      await refresh();
    } catch (e) {
      setNotice(String(e));
    }
  }

  async function handleSyncWatches() {
    if (busy) return;
    setBusy("Syncing folders…");
    setNotice("");
    try {
      const j = await syncWatches();
      setJob(j);
      pollJob(j.job_id);
    } catch (e) {
      setBusy("");
      setNotice(String(e));
    }
  }

  const ready = docs.filter((d) => d.status === "ready").length;
  const hasSearched = searched !== "";
  const activeFilterCount = filterTags.size + scopeDocs.size + (scopeSource ? 1 : 0) + (scopeDocType ? 1 : 0);
  const searchMode = (backend?.search ?? "hybrid").toLowerCase() === "dense" ? "dense" : "hybrid";
  const mode = searchMode === "hybrid" ? "Keywords + meaning" : "Meaning only";
  const rerankOff = ["", "0", "false", "no", "off"].includes((backend?.rerank ?? "on").toLowerCase());
  const rerank = rerankOff ? "Off" : "On";
  // Raw BM25 word-match scores are unbounded, so normalize against the best
  // match in this result set to display them out of 100.
  const sparseMax = Math.max(0, ...hits.map((h) => h.sparse_score ?? 0));
  const jobActive = job !== null && (job.status === "queued" || job.status === "running");
  const jobPct = job && job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;
  const formatHint = (backend?.formats ?? [])
    .map((f) => f.replace(/^\./, ""))
    .filter((f) => !["tif", "tiff", "bmp", "webp", "xlsm", "xltx", "htm", "markdown", "textile", "nfo"].includes(f))
    .join(" / ");
  const sources = [...new Set(docs.map((d) => d.source).filter(Boolean))].sort();

  return (
    <div className="shell">
      <aside className="side">
        <div className="side-brand">
          <span className="logo">P</span>
          <span className="wordmark">procure</span>
          <span className="ver mono">v0.2.1</span>
        </div>

        <div className="side-label">WORKSPACE</div>
        <nav className="side-nav">
          <button className={tab === "search" ? "on" : ""} onClick={() => setTab("search")}>
            <span className="key">1</span> Search
          </button>
          <button className={tab === "library" ? "on" : ""} onClick={() => setTab("library")}>
            <span className="key">2</span> Library
            <span className="count mono">{docs.length}</span>
          </button>
          <button className={tab === "sources" ? "on" : ""} onClick={() => setTab("sources")}>
            <span className="key">3</span> Sources
            {watches.length > 0 && <span className="count mono">{watches.length}</span>}
          </button>
          <button className={tab === "mcp" ? "on" : ""} onClick={() => setTab("mcp")}>
            <span className="key">4</span> MCP Server
            {mcp?.running && <span className="dot" title="MCP server running" />}
          </button>
        </nav>

        <div className="side-label">SEARCH SETUP</div>
        <dl className="side-meta mono">
          <div title="How results are found"><dt>MATCHING</dt><dd>{mode}</dd></div>
          <div title="Whether answer-quality ranking is applied"><dt>ANSWER RANK</dt><dd>{rerank}</dd></div>
          <div><dt>RESULTS</dt><dd>{topK}</dd></div>
          <div><dt>CHUNKS</dt><dd>{backend?.chunks ?? "—"}</dd></div>
          <div><dt>READY</dt><dd>{ready}/{docs.length}</dd></div>
          <div title="OCR backend for scanned PDFs and images"><dt>OCR</dt><dd>{backend?.ocr || "off"}</dd></div>
        </dl>
        <p className="side-hint">Matching + ranking can be changed from the Search tab.</p>

        <div className="side-label">THEME</div>
        <div className="theme-swatches" role="group" aria-label="Color theme">
          {THEMES.map((t) => (
            <button
              key={t.id}
              type="button"
              className={theme === t.id ? "on" : ""}
              onClick={() => setTheme(t.id)}
              aria-pressed={theme === t.id}
              title={t.blurb}
            >
              <span className="swatch" data-swatch={t.id} aria-hidden="true" />
              {t.label}
            </button>
          ))}
        </div>

        <div className="side-foot mono">
          <span>/ focus</span>
          <span>⌘K focus</span>
        </div>
      </aside>

      <div className="main">
        {jobActive && job && (
          <div className="progress" role="status">
            <div className="progress-row mono">
              <span>{job.kind === "watch-sync" ? "SYNC" : `FILE ${job.done + 1}/${job.total}`}</span>
              <span className="truncate">{job.current || job.label}</span>
              <span>{jobPct}%</span>
              <button className="danger" onClick={cancelCurrentJob}>Cancel</button>
            </div>
            <div className="bar"><i style={{ width: `${jobPct}%` }} /></div>
          </div>
        )}

        {(busy || notice) && (
          <p className="status">
            {busy && <span className="busy mono">{busy}</span>}
            {notice && <span>{notice}</span>}
          </p>
        )}

        {tab === "search" ? (
          <main className={hasSearched ? undefined : "search-hero-main"}>
            {hasSearched ? (
              <form className="q" onSubmit={runSearch}>
                <input
                  ref={searchRef}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask across your documents…  ( / to focus )"
                />
                <button type="submit" disabled={!query.trim() || !!busy}>Search</button>
                <button type="button" className="ghost" onClick={clearSearch} title="Clear results and start over">Clear</button>
              </form>
            ) : (
              <div className="hero">
                <h1 className="hero-greet">{greeting}</h1>
                <form className="q hero-q" onSubmit={runSearch}>
                  <input
                    ref={searchRef}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask across your documents…"
                    aria-label="Search your documents"
                  />
                  <button type="submit" disabled={!query.trim() || !!busy} aria-label="Search">→</button>
                </form>
              </div>
            )}

            <div className="filterbar">
              <button
                type="button"
                className={`filtertoggle mono${activeFilterCount ? " on" : ""}`}
                onClick={() => setFiltersOpen((o) => !o)}
                aria-expanded={filtersOpen}
              >
                Filters{activeFilterCount ? ` · ${activeFilterCount}` : ""}
                <span className="arrow">{filtersOpen ? "▾" : "▸"}</span>
              </button>
              <div className={`filterpanel${filtersOpen ? " open" : ""}`}>
                <div className="filterinner">
                  <div className="filterstrip">
                    <label className="ctl">
                      <span>MATCHING</span>
                      <select
                        className="mono"
                        value={searchMode}
                        onChange={(e) => changeSetting({ search: e.target.value })}
                        aria-label="Retrieval matching mode"
                      >
                        <option value="hybrid">Keywords + meaning</option>
                        <option value="dense">Meaning only</option>
                      </select>
                    </label>
                    <label className="ctl">
                      <span>ANSWER RANK</span>
                      <select
                        className="mono"
                        value={rerankOff ? "off" : "on"}
                        onChange={(e) => changeSetting({ rerank: e.target.value })}
                        aria-label="Answer ranking"
                      >
                        <option value="on">On</option>
                        <option value="off">Off</option>
                      </select>
                    </label>
                    <label className="ctl">
                      <span>RESULTS</span>
                      <span className="stepper">
                        <button type="button" onClick={() => setTopK((k) => Math.max(1, k - 1))} aria-label="fewer results">−</button>
                        <code className="mono">{topK}</code>
                        <button type="button" onClick={() => setTopK((k) => Math.min(50, k + 1))} aria-label="more results">+</button>
                      </span>
                    </label>
                    {sources.length > 0 && (
                      <label className="ctl">
                        <span>SOURCE</span>
                        <select
                          className="mono"
                          value={scopeSource}
                          onChange={(e) => setScopeSource(e.target.value)}
                          aria-label="Filter by document source"
                        >
                          <option value="">All</option>
                          {sources.map((s) => (
                            <option key={s} value={s}>{s}</option>
                          ))}
                        </select>
                      </label>
                    )}
                    <label className="ctl">
                      <span>TYPE</span>
                      <select
                        className="mono"
                        value={scopeDocType}
                        onChange={(e) => setScopeDocType(e.target.value)}
                        aria-label="Filter by document type"
                      >
                        <option value="">All</option>
                        <option value="document">Documents</option>
                        <option value="memory">Agent memories</option>
                      </select>
                    </label>
                    {tags.length > 0 && (
                      <span className="chipsgroup" role="group" aria-label="Filter by tag">
                        {tags.map((t) => (
                          <button
                            key={t.tag}
                            type="button"
                            className={`chip mono${filterTags.has(t.tag) ? " on" : ""}`}
                            onClick={() => toggleTag(t.tag)}
                            title={`${t.count} document(s)`}
                          >
                            {t.tag} · {t.count}
                          </button>
                        ))}
                        {filterTags.size > 0 && (
                          <button type="button" className="chip mono clear" onClick={() => setFilterTags(new Set())}>
                            Clear
                          </button>
                        )}
                      </span>
                    )}
                  </div>
                  {docs.length > 0 && (
                    <details className="scope">
                      <summary className="mono">
                        SCOPE: {scopeDocs.size ? `${scopeDocs.size} DOC${scopeDocs.size === 1 ? "" : "S"}` : "ALL DOCUMENTS"}
                      </summary>
                      <div className="scopelist">
                        {docs.map((d) => (
                          <label key={d.doc_id} className="ctl check">
                            <input
                              type="checkbox"
                              checked={scopeDocs.has(d.doc_id)}
                              onChange={() => toggleScopeDoc(d.doc_id)}
                            />
                            <span className="truncate">{d.filename}</span>
                          </label>
                        ))}
                        {scopeDocs.size > 0 && (
                          <button type="button" onClick={() => setScopeDocs(new Set())}>Clear scope</button>
                        )}
                      </div>
                    </details>
                  )}
                </div>
              </div>
            </div>

            {hasSearched && (
              <p className="resultmeta mono">
                {hits.length} RESULT{hits.length === 1 ? "" : "S"} FOR “{searched}”
                {filterTags.size > 0 && ` · TAG: ${[...filterTags].join(", ")}`}
                {scopeSource && ` · SRC: ${scopeSource}`}
                {scopeDocType && ` · TYPE: ${scopeDocType === "memory" ? "memories" : scopeDocType}`}
                {scopeDocs.size > 0 && ` · ${scopeDocs.size} DOC${scopeDocs.size === 1 ? "" : "S"}`}
              </p>
            )}

            <ol className="hits">
              {hits.map((h, i) => (
                <li key={h.chunk_id} className="card">
                  <div className="hithead">
                    <span className="rank mono">#{i + 1}</span>
                    <span className="file">{h.filename}</span>
                    {h.doc_type === "memory" && (
                      <span className="pill mono memory" title="Cross-agent memory">MEMORY</span>
                    )}
                    <span className="score mono" title="Overall similarity to your query">SIMILARITY <b>{fmt(h.score)}</b></span>
                  </div>
                  <p className="snippet">{h.text}</p>
                  <div className="signals mono">
                    <span title="Exact-word match strength, relative to the best match in these results">WORD MATCH <b>{sparseMax > 0 ? fmt((h.sparse_score ?? 0) / sparseMax) : "0%"}</b></span>
                    <span title="How close the meaning is to your query">MEANING <b>{fmt(h.dense_score)}</b></span>
                    <span title="How likely this passage answers your query" className="ans">
                      ANSWER <b>{fmt(h.answerability)}</b>
                      <span className="minibar">
                        <i style={{ width: `${Math.max(0, Math.min(1, h.answerability ?? 0)) * 100}%` }} />
                      </span>
                    </span>
                    {(h.entailment ?? 0) >= 0.5 && (
                      <span className="direct" title="This passage directly supports an answer to your query">
                        DIRECT ANSWER
                      </span>
                    )}
                  </div>
                  <div className="docline mono">DOC {h.doc_id} · CHUNK {h.chunk_id}</div>
                </li>
              ))}
            </ol>
            {hasSearched && !hits.length && !busy && (
              <p className="empty">No matches — try different words or fewer filters. Add files from Sources.</p>
            )}
          </main>
        ) : tab === "library" ? (
          <main>
            <div className="libbar">
              <span className="mono">{selected.size} SELECTED</span>
              {selected.size > 0 && (
                <button
                  className="danger"
                  onClick={async () => {
                    setBusy("Deleting…");
                    try {
                      for (const id of selected) await deleteDoc(id);
                      setSelected(new Set());
                      await refresh();
                    } catch (e) {
                      setNotice(String(e));
                    } finally {
                      setBusy("");
                    }
                  }}
                >
                  Delete selected
                </button>
              )}
            </div>
            <table className="lib">
              <thead>
                <tr>
                  <th></th>
                  <th>File</th>
                  <th>Status</th>
                  <th>Chunks</th>
                  <th>Tags</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.doc_id} className={selected.has(d.doc_id) ? "sel" : ""}>
                    <td>
                      <input
                        type="checkbox"
                        checked={selected.has(d.doc_id)}
                        onChange={() => toggle(d.doc_id)}
                        aria-label={`select ${d.filename}`}
                      />
                    </td>
                    <td className="file">{d.filename}<span className="docid mono">{d.doc_id} · {d.source || "upload"}{(d.doc_type || "document") === "memory" ? " · memory" : ""}</span></td>
                    <td>
                      <span className={`pill mono ${d.status}`}>{d.status.toUpperCase()}</span>
                      {d.error && <span className="err"> — {d.error}</span>}
                    </td>
                    <td className="mono">{d.chunk_count}</td>
                    <td className="tagcell">
                      {editingTags === d.doc_id ? (
                        <span className="tagedit">
                          <input
                            className="mono"
                            value={tagDraft}
                            autoFocus
                            placeholder="comma, separated"
                            onChange={(e) => setTagDraft(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") saveTags(d.doc_id);
                              if (e.key === "Escape") setEditingTags(null);
                            }}
                            aria-label={`tags for ${d.filename}`}
                          />
                          <button onClick={() => saveTags(d.doc_id)}>Save</button>
                          <button onClick={() => setEditingTags(null)}>✕</button>
                        </span>
                      ) : (
                        <span
                          className="tagview"
                          onClick={() => { setEditingTags(d.doc_id); setTagDraft((d.tags ?? []).join(", ")); }}
                          title="Click to edit tags"
                          role="button"
                          tabIndex={0}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") { setEditingTags(d.doc_id); setTagDraft((d.tags ?? []).join(", ")); }
                          }}
                        >
                          {(d.tags ?? []).length
                            ? (d.tags ?? []).map((t) => <span key={t} className="chip mono sm">{t}</span>)
                            : <span className="mono dim">+ tag</span>}
                        </span>
                      )}
                    </td>
                    <td className="acts">
                      <button onClick={() => doReingest(d.doc_id)}>Re-ingest</button>
                      <button className="danger" onClick={() => doDelete(d.doc_id)}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!docs.length && <p className="empty">Library is empty — add files from Sources.</p>}
          </main>
        ) : tab === "sources" ? (
          <main>
            <div className="resultmeta mono">ADD FILES</div>
            <section
              className={`drop tab-sources-drop${drag ? " over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); handleFiles(e.dataTransfer.files); }}
              onClick={() => fileRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === "Enter") fileRef.current?.click(); }}
            >
              <input
                ref={fileRef}
                type="file"
                multiple
                hidden
                onChange={(e) => {
                  if (e.target.files) handleFiles(e.target.files);
                  e.target.value = "";
                }}
              />
              <strong>{drag ? "DROP TO INGEST" : "DRAG + DROP FILES TO INGEST"}</strong>
              <span className="mono">{formatHint || "pdf / docx / md / txt"} — or click to browse</span>
            </section>

            <div className="resultmeta mono">ADD FROM URL</div>
            <form className="urlform" onSubmit={handleAddUrl}>
              <input
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/article"
                aria-label="URL to ingest"
              />
              <input
                className="mono"
                value={urlTags}
                onChange={(e) => setUrlTags(e.target.value)}
                placeholder="tags, comma separated"
                aria-label="Tags for this URL"
              />
              <button type="submit" className="primary" disabled={!url.trim() || !!busy}>Add URL</button>
            </form>

            <div className="resultmeta mono">WATCHED FOLDERS</div>
            <form className="urlform" onSubmit={handleAddWatch}>
              <input
                className="mono"
                value={watchPath}
                onChange={(e) => setWatchPath(e.target.value)}
                placeholder="/absolute/path/to/folder"
                aria-label="Folder path to watch"
              />
              <label className="ctl check">
                <input
                  type="checkbox"
                  checked={watchRecursive}
                  onChange={(e) => setWatchRecursive(e.target.checked)}
                />
                Recursive
              </label>
              <button type="submit" disabled={!watchPath.trim() || !!busy}>Watch</button>
              <button type="button" onClick={handleSyncWatches} disabled={!watches.length || !!busy}>
                Sync now
              </button>
            </form>
            {lastScan && <p className="side-hint mono">Last scan: {lastScan}</p>}
            {watches.length > 0 ? (
              <table className="lib">
                <thead>
                  <tr>
                    <th>Path</th>
                    <th>Recursive</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {watches.map((w) => (
                    <tr key={w.id}>
                      <td className="mono">{w.path}</td>
                      <td className="mono">{w.recursive ? "yes" : "no"}</td>
                      <td className="acts">
                        <button className="danger" onClick={() => handleRemoveWatch(w.id)}>Remove</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="empty">No watched folders. Add one above — new files ingest automatically, edits re-ingest in place, deletions mark docs missing.</p>
            )}
          </main>
        ) : (
          <main>
            <div className="mcpbar">
              <span className={`pill mono ${mcp?.running ? "ready" : ""}`}>
                {mcp?.running ? "RUNNING" : "STOPPED"}
              </span>
              {mcp?.running && mcp.pid !== null && (
                <span className="mono dim">PID {mcp.pid} · UP {fmtUptime(mcp.uptime_s)}</span>
              )}
              {!mcp?.running && mcp?.exit_code !== null && mcp?.exit_code !== undefined && (
                <span className="mono dim">LAST EXIT {mcp.exit_code} — SEE LOG</span>
              )}
              <button
                className={mcp?.running ? "danger" : "primary"}
                onClick={toggleMcp}
                disabled={!!busy}
              >
                {mcp?.running ? "Stop server" : "Start server"}
              </button>
            </div>

            <div className="mcpurl">
              <input
                className="mono"
                readOnly
                value={mcp?.url ?? ""}
                placeholder="Server URL appears here"
                onFocus={(e) => e.target.select()}
                aria-label="MCP server URL"
              />
              <button onClick={copyMcpUrl} disabled={!mcp}>
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
            <p className="side-hint">
              Register this URL as a Streamable HTTP server in your harness
              (e.g. <code className="mono">Muse mcp add --transport http procure {mcp?.url ?? "…"}</code>).
              Same library, embeddings, and search pipeline as this workspace.
            </p>

            <div className="mcptools">
              <div className="mcptool">
                <code className="mono">procure_search</code>
                <span>Ask across the library — hybrid BM25 + dense retrieval, reranked by answerability. Scopes to doc IDs, tags, and type (documents vs agent memories).</span>
              </div>
              <div className="mcptool">
                <code className="mono">procure_add_text</code>
                <span>Save transcripts, outputs, notes — chunked, embedded, searchable immediately. Accepts tags; pass type memory for cross-agent memories.</span>
              </div>
              <div className="mcptool">
                <code className="mono">procure_add_url</code>
                <span>Fetch a web page into the library — article text extracted and searchable immediately.</span>
              </div>
              <div className="mcptool">
                <code className="mono">procure_list_documents</code>
                <span>See what is stored — every document with status, tags, and chunk counts.</span>
              </div>
              <div className="mcptool">
                <code className="mono">procure://documents/{"{doc_id}"}</code>
                <span>Fetch the full text of a document found via search or listing.</span>
              </div>
              <div className="mcptool">
                <code className="mono">procure://guide</code>
                <span>Agent usage guide — how to search, read documents, and store memories. Same content as the skill below.</span>
              </div>
            </div>

            <div className="resultmeta mono">AGENT SKILL{skill?.version ? ` · v${skill.version}` : ""}</div>
            <p className="side-hint">
              Install the procure skill into your agent harnesses so agents proactively
              search the library and save memories. Existing installs are left alone;
              reinstall to pick up skill updates.
            </p>
            {skill && (
              <table className="lib">
                <thead>
                  <tr>
                    <th>Harness</th>
                    <th>Location</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {skill.targets.map((t) => (
                    <tr key={t.id}>
                      <td>{t.label}{t.detected ? "" : <span className="mono dim"> (not detected)</span>}</td>
                      <td className="mono">{t.path}</td>
                      <td>
                        {!t.installed && <span className="mono dim">not installed</span>}
                        {t.installed && !t.outdated && (
                          <span className="pill mono ready">v{t.installed_version || "?"}</span>
                        )}
                        {t.installed && t.outdated && (
                          <span className="pill mono missing">v{t.installed_version || "?"} → v{t.bundled_version || "?"}</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <div className="mcpbar">
              <button className="primary" onClick={() => doInstallSkill(false)} disabled={!!busy}>
                Install skill
              </button>
              <button onClick={() => doInstallSkill(true)} disabled={!!busy}>
                Reinstall
              </button>
            </div>

            <div className="resultmeta mono">SERVER OUTPUT</div>
            <pre
              ref={mcpLogRef}
              className="mcplog mono"
              onScroll={(e) => {
                const el = e.currentTarget;
                mcpFollow.current =
                  el.scrollHeight - el.scrollTop - el.clientHeight < 24;
              }}
            >
              {mcpLines.length
                ? mcpLines.map((l) => l.text).join("\n")
                : "Server output appears here. Start the server to see it."}
            </pre>
          </main>
        )}
      </div>
    </div>
  );
}
