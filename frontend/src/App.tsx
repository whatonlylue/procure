import { useCallback, useEffect, useRef, useState } from "react";
import {
  Doc,
  Health,
  Hit,
  Job,
  McpLogLine,
  McpStatus,
  McpTool,
  SkillStatus,
  TagCount,
  cancelJob,
  deleteDoc,
  exportLibrary,
  getJob,
  health as fetchHealth,
  importLibrary,
  installSkills,
  libraryVersion,
  listDocs,
  listTags,
  mcpLogs,
  mcpStart,
  mcpStatus,
  mcpStop,
  mcpTools,
  patchDoc,
  reingestDoc,
  rerankIsOff,
  search,
  setDocTags,
  skillsStatus,
  updateSettings,
  uploadFilesAsync,
  workspaceLogs,
} from "./api";
import { THEMES, ThemeName, applyTheme, getInitialTheme } from "./theme";
import { Toast, ToastKind, ToastStack } from "./toasts";

type Tab = "search" | "library" | "sources" | "mcp";

const TABS: { id: Tab; label: string; key: string }[] = [
  { id: "search", label: "Search", key: "1" },
  { id: "library", label: "Library", key: "2" },
  { id: "sources", label: "Sources", key: "3" },
  { id: "mcp", label: "MCP Server", key: "4" },
];

const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]);
const LIB_PAGE_SIZE = 50;
// Cheap revision poll: two aggregate SQLite queries, no vector access.
const LIB_VERSION_POLL_MS = 5000;
const SCOPE_FETCH_MAX = 500;
const SCOPE_RENDER_MAX = 200;

function fmt(n: number | undefined): string {
  if (typeof n !== "number" || !Number.isFinite(n)) return "—";
  return `${Math.round(Math.max(0, Math.min(1, n)) * 100)}%`;
}

function parseTags(s: string): string[] {
  return s.split(",").map((t) => t.trim()).filter(Boolean);
}

function toggleInSet(setter: (f: (s: Set<string>) => Set<string>) => void, id: string) {
  setter((s) => {
    const n = new Set(s);
    if (n.has(id)) n.delete(id);
    else n.add(id);
    return n;
  });
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
  const [docsTotal, setDocsTotal] = useState(0);
  const [scopeList, setScopeList] = useState<Doc[]>([]);
  const [scopeTruncated, setScopeTruncated] = useState(false);
  const [tags, setTags] = useState<TagCount[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<Hit[]>([]);
  const [searched, setSearched] = useState("");
  const [busy, setBusy] = useState("");
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [drag, setDrag] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [topK, setTopK] = useState(10);
  const [backend, setBackend] = useState<Health | null>(null);
  const [mcp, setMcp] = useState<McpStatus | null>(null);
  const [mcpLines, setMcpLines] = useState<McpLogLine[]>([]);
  const [wsLines, setWsLines] = useState<McpLogLine[]>([]);
  const [tools, setTools] = useState<McpTool[]>([]);
  const [copied, setCopied] = useState(false);
  const [skill, setSkill] = useState<SkillStatus | null>(null);
  // Search scope.
  const [filterTags, setFilterTags] = useState<Set<string>>(new Set());
  const [scopeDocs, setScopeDocs] = useState<Set<string>>(new Set());
  const [scopeFilter, setScopeFilter] = useState("");
  const [scopeDocType, setScopeDocType] = useState("");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [greeting] = useState<string>(pickGreeting);
  // Library browser.
  const [editingTags, setEditingTags] = useState<string | null>(null);
  const [tagDraft, setTagDraft] = useState("");
  const [libQuery, setLibQuery] = useState("");
  const [libSort, setLibSort] = useState("newest");
  const [libType, setLibType] = useState("");
  const [libPage, setLibPage] = useState(0);
  // Sources tab.
  const [uploadDocType, setUploadDocType] = useState("document");
  const fileRef = useRef<HTMLInputElement>(null);
  const importRef = useRef<HTMLInputElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const mcpNext = useRef(0);
  const wsNext = useRef(0);
  const mcpLogRef = useRef<HTMLPreElement>(null);
  const wsLogRef = useRef<HTMLPreElement>(null);
  const mcpFollow = useRef(true);
  const wsFollow = useRef(true);
  const pollRef = useRef<number | null>(null);
  const libVersion = useRef<string | null>(null);
  const refreshing = useRef(false);
  const toastId = useRef(0);

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // User alerts (errors, statuses, confirmations) surface as dismissable
  // toasts in the top-right corner instead of inline text.
  const notify = useCallback((text: string, kind: ToastKind = "info") => {
    const trimmed = text.trim();
    if (!trimmed) return;
    toastId.current += 1;
    const id = toastId.current;
    setToasts((prev) => [...prev.slice(-4), { id, kind, text: trimmed }]);
  }, []);

  const notifyError = useCallback(
    (e: unknown) => {
      notify(String(e), "error");
    },
    [notify]
  );

  // Compatibility bridge: every former inline notice now surfaces as a
  // dismissable toast. Empty clears are no-ops; error-looking text gets
  // the error style, everything else the info style.
  const setNotice = (text: string) => {
    const t = text.trim();
    if (!t) return;
    notify(t, /fail|error/i.test(t) ? "error" : "info");
  };

  const refresh = useCallback(async () => {
    refreshing.current = true;
    try {
      const [lib, scope, h, m, t, v] = await Promise.all([
        listDocs({ query: libQuery || undefined, sort: libSort, docType: libType || undefined, limit: LIB_PAGE_SIZE, offset: libPage * LIB_PAGE_SIZE }),
        listDocs({ limit: SCOPE_FETCH_MAX, sort: "newest" }),
        fetchHealth(),
        mcpStatus(),
        listTags(),
        libraryVersion(),
      ]);
      libVersion.current = JSON.stringify(v);
      setDocs(lib.documents);
      setDocsTotal(lib.total);
      setScopeList(scope.documents);
      setScopeTruncated(scope.total > scope.documents.length);
      setBackend(h);
      setMcp(m);
      setTags(t);
      // Prune filters that no longer resolve: vanished tags and deleted
      // docs used to leave invisible filters (or dead scope ids) that
      // silently returned nothing.
      const tagNames = new Set(t.map((x) => x.tag));
      setFilterTags((prev) => new Set([...prev].filter((x) => tagNames.has(x))));
      const ids = new Set(scope.documents.map((d) => d.doc_id));
      setScopeDocs((prev) => new Set([...prev].filter((x) => ids.has(x))));
      setSelected((prev) => new Set([...prev].filter((x) => ids.has(x))));
    } catch (e) {
      setNotice(String(e));
    } finally {
      refreshing.current = false;
    }
  }, [libQuery, libSort, libType, libPage]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;

  // External writers (MCP server memories, watch sync) bypass the UI, so
  // poll the cheap version fingerprint and reload only when it moves.
  // Failures stay silent: the next tick retries, and a failed reload
  // already surfaces through refresh()'s own error toast.
  const checkVersion = useCallback(async () => {
    if (document.visibilityState === "hidden" || refreshing.current) return;
    let key: string;
    try {
      key = JSON.stringify(await libraryVersion());
    } catch {
      return;
    }
    if (libVersion.current === null) {
      libVersion.current = key;
      return;
    }
    if (key !== libVersion.current) {
      // refresh() records the post-reload fingerprint itself.
      await refreshRef.current();
    }
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => {
      void checkVersion();
    }, LIB_VERSION_POLL_MS);
    const onFocus = () => {
      void checkVersion();
    };
    window.addEventListener("focus", onFocus);
    return () => {
      window.clearInterval(id);
      window.removeEventListener("focus", onFocus);
    };
  }, [checkVersion]);

  // Re-check as soon as the Library tab opens.
  useEffect(() => {
    if (tab === "library") void checkVersion();
  }, [tab, checkVersion]);

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
        const [s, l, wl] = await Promise.all([
          mcpStatus(),
          mcpLogs(mcpNext.current),
          workspaceLogs(wsNext.current),
        ]);
        if (!alive) return;
        setMcp(s);
        mcpNext.current = l.next;
        if (l.lines.length) {
          setMcpLines((prev) => [...prev, ...l.lines].slice(-1000));
        }
        wsNext.current = wl.next;
        if (wl.lines.length) {
          setWsLines((prev) => [...prev, ...wl.lines].slice(-1000));
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

  // Follow the log tails unless the user scrolled up.
  useEffect(() => {
    const el = mcpLogRef.current;
    if (el && mcpFollow.current) el.scrollTop = el.scrollHeight;
  }, [mcpLines]);
  useEffect(() => {
    const el = wsLogRef.current;
    if (el && wsFollow.current) el.scrollTop = el.scrollHeight;
  }, [wsLines]);

  // Skill install state + tool list, refreshed whenever the MCP tab opens.
  useEffect(() => {
    if (tab !== "mcp") return;
    let alive = true;
    skillsStatus()
      .then((s) => { if (alive) setSkill(s); })
      .catch((e) => { if (alive) setNotice(String(e)); });
    mcpTools()
      .then((tl) => { if (alive) setTools(tl); })
      .catch(() => { /* tools block stays empty */ });
    return () => { alive = false; };
  }, [tab]);

  // Keyboard: "/" or Cmd/Ctrl+K focuses search; 1-4 switch tabs.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null;
      const typing = !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT");
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setTab("search");
        searchRef.current?.focus();
        searchRef.current?.select();
      } else if (e.key === "/" && !typing) {
        e.preventDefault();
        setTab("search");
        searchRef.current?.focus();
      } else if (!typing && !e.metaKey && !e.ctrlKey && !e.altKey) {
        const found = TABS.find((x) => x.key === e.key);
        if (found) setTab(found.id);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function summarizeResults(res: unknown): string {
    const docs = (Array.isArray(res) ? res : [res]) as { status?: string; filename?: string }[];
    const ready = docs.filter((d) => d.status === "ready").length;
    const dup = docs.filter((d) => d.status === "duplicate").length;
    const failed = docs.filter((d) => d.status === "failed");
    const cancelled = docs.filter((d) => d.status === "cancelled").length;
    if (!ready && !dup && !cancelled && !failed.length) return "Done.";
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
          if (j.status === "failed") {
            notifyError(`Job failed: ${j.error}`);
          } else {
            const summary = summarizeResults(j.result);
            notify(summary, /fail|error/i.test(summary) ? "error" : "success");
          }
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
    try {
      const j = await uploadFilesAsync(arr, uploadDocType);
      setJob(j);
      pollJob(j.job_id);
    } catch (e) {
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
    setNotice("");
    try {
      const j = await reingestDoc(id, true);
      setJob(j as Job);
      pollJob((j as Job).job_id);
    } catch (e) {
      setNotice(String(e));
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
      if (errs.length) {
        notifyError(
          `Skill install errors: ${errs.map((r) => `${r.id}: ${r.detail}`).join("; ")}`
        );
      } else {
        notify(
          `Skill ${results.map((r) => `${r.id} ${r.action}`).join(", ")}.`,
          "success"
        );
      }
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

  async function changeSetting(patch: { search?: string; rerank?: string; mcp_autostart?: boolean }) {
    setNotice("");
    try {
      const s = await updateSettings(patch);
      setBackend((b) => (b ? { ...b, search: s.search, rerank: s.rerank, mcp_autostart: s.mcp_autostart } : b));
      if ((patch.search !== undefined || patch.rerank !== undefined) && mcp?.running) {
        setNotice("Retrieval settings saved — restart the MCP server to apply them there.");
      }
    } catch (e) {
      setNotice(String(e));
    }
  }

  async function handleExport() {
    setNotice("");
    try {
      const blob = await exportLibrary();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "procure-export.json";
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      setNotice(String(e));
    }
  }

  async function handleImport(files: FileList | null) {
    if (!files || !files.length) return;
    setNotice("");
    try {
      const j = await importLibrary(files[0]);
      setJob(j);
      pollJob(j.job_id);
    } catch (e) {
      setNotice(String(e));
    }
  }

  async function handleRenameType(id: string, patch: { filename?: string; doc_type?: string }) {
    setNotice("");
    try {
      await patchDoc(id, patch);
      await refresh();
    } catch (e) {
      setNotice(String(e));
    }
  }

  const ready = scopeList.filter((d) => d.status === "ready").length;
  const hasSearched = searched !== "";
  const activeFilterCount = filterTags.size + scopeDocs.size + (scopeDocType ? 1 : 0);
  const searchMode = (backend?.search ?? "hybrid").toLowerCase() === "dense" ? "dense" : "hybrid";
  const mode = searchMode === "hybrid" ? "Keywords + meaning" : "Meaning only";
  const off = rerankIsOff(backend?.rerank);
  const neural = (backend?.rerank_backend ?? "heuristic") === "neural";
  const rerank = off ? "Off" : neural ? "On" : "On · words";
  // Raw BM25 word-match scores are unbounded, so normalize against the best
  // match in this result set to display them out of 100.
  const sparseMax = Math.max(0, ...hits.map((h) => h.sparse_score ?? 0));
  const jobActive = job !== null && (job.status === "queued" || job.status === "running");
  const jobPct = job && job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;
  const ocrOn = !!(backend?.ocr && backend.ocr !== "off");
  const formatHint = (backend?.formats ?? [])
    .map((f) => f.replace(/^\./, ""))
    .filter((f) => {
      if (IMAGE_EXTS.has(f)) return ocrOn;
      return !["tif", "tiff", "bmp", "webp", "xlsm", "xltx", "htm", "markdown", "textile", "nfo"].includes(f);
    })
    .join(" / ");
  const scopeShown = scopeFilter
    ? scopeList.filter((d) => d.filename.toLowerCase().includes(scopeFilter.toLowerCase()))
    : scopeList;
  const scopeCapped = scopeShown.slice(0, SCOPE_RENDER_MAX);
  const libPages = Math.max(1, Math.ceil(docsTotal / LIB_PAGE_SIZE));

  return (
    <div className="shell">
      <ToastStack toasts={toasts} onDismiss={dismissToast} />
      <aside className="side">
        <div className="side-brand">
          <span className="logo">P</span>
          <span className="wordmark">procure</span>
          <span className="ver mono">v{backend?.version ?? "…"}</span>
        </div>

        <div className="side-label">WORKSPACE</div>
        <nav className="side-nav">
          {TABS.map((t) => (
            <button key={t.id} className={tab === t.id ? "on" : ""} onClick={() => setTab(t.id)}>
              <span className="key">{t.key}</span> {t.label}
              {t.id === "library" && <span className="count mono">{docsTotal}</span>}
              {t.id === "mcp" && mcp?.running && <span className="dot" title="MCP server running" />}
            </button>
          ))}
        </nav>

        <div className="side-label">SEARCH SETUP</div>
        <dl className="side-meta mono">
          <div title="How results are found"><dt>MATCHING</dt><dd>{mode}</dd></div>
          <div title={neural ? "Neural cross-encoder + NLI when loaded, word-match coverage otherwise" : "Word-match coverage (no neural models in this build)"}><dt>ANSWER RANK</dt><dd>{rerank}</dd></div>
          <div><dt>RESULTS</dt><dd>{topK}</dd></div>
          <div><dt>CHUNKS</dt><dd>{backend?.chunks ?? "—"}</dd></div>
          <div><dt>READY</dt><dd>{ready}{scopeTruncated ? "+" : ""}/{docsTotal}</dd></div>
          <div title="OCR backend for scanned PDFs and images"><dt>OCR</dt><dd>{backend?.ocr || "off"}</dd></div>
        </dl>
        <p className="side-hint">Matching + ranking can be changed from the Search tab.</p>
        {backend?.dense && !backend.dense.dense_ok && (
          <p className="warn mono" title={backend.dense.warning}>
            DENSE DEGRADED — {backend.dense.warning}
          </p>
        )}

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
              <span>
                {job.total > 0 ? `FILE ${Math.min(Math.floor(job.done) + 1, job.total)}/${job.total}` : job.label.toUpperCase()}
              </span>
              <span className="truncate">{job.current || job.label}</span>
              <span>{job.total > 0 ? `${jobPct}%` : "…"}</span>
              <button className="danger" onClick={cancelCurrentJob}>Cancel</button>
            </div>
            <div className="bar"><i style={{ width: `${jobPct}%` }} /></div>
          </div>
        )}

        {busy && (
          <p className="status">
            <span className="busy mono">{busy}</span>
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
                        value={off ? "off" : "on"}
                        onChange={(e) => changeSetting({ rerank: e.target.value })}
                        aria-label="Answer ranking"
                      >
                        <option value="on">On{neural ? "" : " (words)"}</option>
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
                  </div>
                  {tags.length > 0 && (
                    <details className="scope">
                      <summary className="mono">
                        TAGS: {filterTags.size ? `${filterTags.size} SELECTED` : "ALL"}
                      </summary>
                      <div className="taglist" role="group" aria-label="Filter by tag">
                        {tags.map((t) => (
                          <button
                            key={t.tag}
                            type="button"
                            className={`chip mono${filterTags.has(t.tag) ? " on" : ""}`}
                            onClick={() => toggleInSet(setFilterTags, t.tag)}
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
                      </div>
                    </details>
                  )}
                  {scopeList.length > 0 && (
                    <details className="scope">
                      <summary className="mono">
                        SCOPE: {scopeDocs.size ? `${scopeDocs.size} DOC${scopeDocs.size === 1 ? "" : "S"}` : "ALL DOCUMENTS"}
                      </summary>
                      <div className="scopelist">
                        <input
                          className="mono"
                          value={scopeFilter}
                          onChange={(e) => setScopeFilter(e.target.value)}
                          placeholder="filter documents…"
                          aria-label="Filter scoped documents"
                        />
                        {scopeCapped.map((d) => (
                          <label key={d.doc_id} className="ctl check">
                            <input
                              type="checkbox"
                              checked={scopeDocs.has(d.doc_id)}
                              onChange={() => toggleInSet(setScopeDocs, d.doc_id)}
                            />
                            <span className="truncate">{d.filename}</span>
                          </label>
                        ))}
                        {scopeShown.length > scopeCapped.length && (
                          <span className="mono dim">…{scopeShown.length - scopeCapped.length} more — refine the filter</span>
                        )}
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
                    <span className="score mono" title="Blended rank score for your query">SCORE <b>{fmt(h.score)}</b></span>
                  </div>
                  <p className="snippet">{h.text}</p>
                  <div className="signals mono">
                    <span title="Exact-word match strength, relative to the best match in these results">WORD MATCH <b>{sparseMax > 0 ? fmt((h.sparse_score ?? 0) / sparseMax) : "0%"}</b></span>
                    <span title="How close the meaning is to your query">MEANING <b>{fmt(h.dense_score)}</b></span>
                    {!off && (
                      <span title="How likely this passage answers your query" className="ans">
                        ANSWER <b>{fmt(h.answerability)}</b>
                        <span className="minibar">
                          <i style={{ width: `${Math.max(0, Math.min(1, h.answerability ?? 0)) * 100}%` }} />
                        </span>
                      </span>
                    )}
                    {!off && (h.entailment ?? 0) >= 0.5 && (
                      <span className="direct" title="This passage directly supports an answer to your query">
                        DIRECT ANSWER
                      </span>
                    )}
                  </div>
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
              <input
                className="mono"
                value={libQuery}
                onChange={(e) => { setLibQuery(e.target.value); setLibPage(0); }}
                placeholder="search library…"
                aria-label="Search library"
              />
              <select
                className="mono"
                value={libSort}
                onChange={(e) => { setLibSort(e.target.value); setLibPage(0); }}
                aria-label="Sort library"
              >
                <option value="newest">Newest</option>
                <option value="oldest">Oldest</option>
                <option value="name">Name</option>
              </select>
              <select
                className="mono"
                value={libType}
                onChange={(e) => { setLibType(e.target.value); setLibPage(0); }}
                aria-label="Filter library by type"
              >
                <option value="">All types</option>
                <option value="document">Documents</option>
                <option value="memory">Memories</option>
              </select>
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
              <button onClick={handleExport} title="Download a JSON backup of the library">Export</button>
              <button onClick={() => importRef.current?.click()} title="Restore from an export file">Import</button>
              <input
                ref={importRef}
                type="file"
                accept="application/json,.json"
                hidden
                onChange={(e) => {
                  handleImport(e.target.files);
                  e.target.value = "";
                }}
              />
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
                        onChange={() => toggleInSet(setSelected, d.doc_id)}
                        aria-label={`select ${d.filename}`}
                      />
                    </td>
                    <td className="file">{d.filename}<span className="docid mono">{d.doc_id}{(d.doc_type || "document") === "memory" ? " · memory" : ""}</span></td>
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
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              setEditingTags(d.doc_id); setTagDraft((d.tags ?? []).join(", "));
                            }
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
                      {(d.doc_type || "document") === "memory" ? (
                        <button onClick={() => handleRenameType(d.doc_id, { doc_type: "document" })} title="Convert to a regular document">Unmark memory</button>
                      ) : (
                        <button onClick={() => handleRenameType(d.doc_id, { doc_type: "memory" })} title="Convert to a cross-agent memory">Mark memory</button>
                      )}
                      <button className="danger" onClick={() => doDelete(d.doc_id)}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="libbar">
              <button disabled={libPage <= 0} onClick={() => setLibPage((p) => Math.max(0, p - 1))}>← Prev</button>
              <span className="mono">PAGE {libPage + 1}/{libPages} · {docsTotal} DOCS</span>
              <button disabled={libPage + 1 >= libPages} onClick={() => setLibPage((p) => p + 1)}>Next →</button>
            </div>
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
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") fileRef.current?.click(); }}
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
            <div className="urlform">
              <label className="ctl">
                <span>TYPE</span>
                <select
                  className="mono"
                  value={uploadDocType}
                  onChange={(e) => setUploadDocType(e.target.value)}
                  aria-label="Type for uploaded files"
                >
                  <option value="document">Documents</option>
                  <option value="memory">Memories</option>
                </select>
              </label>
            </div>

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
            <label className="ctl check">
              <input
                type="checkbox"
                checked={!!backend?.mcp_autostart}
                onChange={(e) => changeSetting({ mcp_autostart: e.target.checked })}
              />
              Start automatically with the workspace
            </label>

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

            <details className="mcptools">
              <summary className="mono">
                COMMANDS: {tools.length ? `${tools.length} AVAILABLE` : "…"}
              </summary>
              {tools.length ? tools.map((t) => (
                <div className="mcptool" key={t.name}>
                  <code className="mono">{t.name}</code>
                  <span>{t.description}</span>
                </div>
              )) : (
                <div className="mcptool"><span className="mono dim">Loading tool list…</span></div>
              )}
            </details>

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

            <div className="resultmeta mono">MCP SERVER OUTPUT</div>
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

            <div className="resultmeta mono">WORKSPACE SERVER LOG</div>
            <pre
              ref={wsLogRef}
              className="mcplog mono"
              onScroll={(e) => {
                const el = e.currentTarget;
                wsFollow.current =
                  el.scrollHeight - el.scrollTop - el.clientHeight < 24;
              }}
            >
              {wsLines.length
                ? wsLines.map((l) => l.text).join("\n")
                : "Workspace log appears here."}
            </pre>
          </main>
        )}
      </div>
    </div>
  );
}
