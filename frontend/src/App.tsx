import { useCallback, useEffect, useRef, useState } from "react";
import {
  Doc,
  Health,
  Hit,
  McpLogLine,
  McpStatus,
  deleteDoc,
  health as fetchHealth,
  listDocs,
  mcpLogs,
  mcpStart,
  mcpStatus,
  mcpStop,
  reingestDoc,
  search,
  updateSettings,
  uploadFiles,
} from "./api";

type Tab = "search" | "library" | "mcp";

interface UploadState {
  done: number;
  total: number;
  current: string;
  failed: string[];
}

function fmt(n: number | undefined): string {
  return typeof n === "number" && Number.isFinite(n) ? `${Math.round(n * 100)}%` : "—";
}

export default function App() {
  const [tab, setTab] = useState<Tab>("search");
  const [docs, setDocs] = useState<Doc[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<Hit[]>([]);
  const [searched, setSearched] = useState("");
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [drag, setDrag] = useState(false);
  const [upload, setUpload] = useState<UploadState | null>(null);
  const [topK, setTopK] = useState(10);
  const [backend, setBackend] = useState<Health | null>(null);
  const [mcp, setMcp] = useState<McpStatus | null>(null);
  const [mcpLines, setMcpLines] = useState<McpLogLine[]>([]);
  const [copied, setCopied] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const mcpNext = useRef(0);
  const mcpLogRef = useRef<HTMLPreElement>(null);
  const mcpFollow = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const [d, h, m] = await Promise.all([listDocs(), fetchHealth(), mcpStatus()]);
      setDocs(d);
      setBackend(h);
      setMcp(m);
    } catch (e) {
      setNotice(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

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

  async function handleFiles(files: FileList | File[]) {
    const arr = Array.from(files);
    if (!arr.length) return;
    setNotice("");
    setUpload({ done: 0, total: arr.length, current: arr[0].name, failed: [] });
    const failed: string[] = [];
    let ok = 0;
    // Sequential per-file POSTs so progress is visible per file.
    for (let i = 0; i < arr.length; i++) {
      setUpload({ done: i, total: arr.length, current: arr[i].name, failed: [...failed] });
      setBusy(`Uploading ${i + 1}/${arr.length} — ${arr[i].name}`);
      try {
        await uploadFiles([arr[i]]);
        ok++;
      } catch {
        failed.push(arr[i].name);
      }
    }
    setUpload({ done: arr.length, total: arr.length, current: "", failed });
    setBusy("");
    setNotice(
      failed.length
        ? `Ingested ${ok}/${arr.length}. Failed: ${failed.join(", ")}`
        : `Ingested ${ok} file(s).`
    );
    await refresh();
  }

  async function runSearch(e?: React.FormEvent) {
    e?.preventDefault();
    if (!query.trim() || busy) return;
    setBusy("Searching…");
    setNotice("");
    try {
      const res = await search(query, topK);
      setHits(res);
      setSearched(query);
    } catch (err) {
      setNotice(String(err));
    } finally {
      setBusy("");
    }
  }

  function toggle(id: string) {
    setSelected((s) => {
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

  const ready = docs.filter((d) => d.status === "ready").length;
  const searchMode = (backend?.search ?? "hybrid").toLowerCase() === "dense" ? "dense" : "hybrid";
  const mode = searchMode === "hybrid" ? "Keywords + meaning" : "Meaning only";
  const rerankOff = ["", "0", "false", "no", "off"].includes((backend?.rerank ?? "on").toLowerCase());
  const rerank = rerankOff ? "Off" : "On";
  // Raw BM25 word-match scores are unbounded, so normalize against the best
  // match in this result set to display them out of 100.
  const sparseMax = Math.max(0, ...hits.map((h) => h.sparse_score ?? 0));

  return (
    <div className="shell">
      <aside className="side">
        <div className="side-brand">
          <span className="logo">P</span>
          <span className="wordmark">procure</span>
          <span className="ver mono">v0.1</span>
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
          <button className={tab === "mcp" ? "on" : ""} onClick={() => setTab("mcp")}>
            <span className="key">3</span> MCP Server
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
        </dl>
        <p className="side-hint">Matching + ranking can be changed from the Search tab.</p>

        <div className="side-foot mono">
          <span>/ focus</span>
          <span>⌘K focus</span>
        </div>
      </aside>

      <div className="main">
        <section
          className={`drop${drag ? " over" : ""}`}
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
          <span className="mono">pdf / docx / pptx / md / txt — or click to browse</span>
        </section>

        {upload && upload.done < upload.total && (
          <div className="progress" role="status">
            <div className="progress-row mono">
              <span>FILE {upload.done + 1}/{upload.total}</span>
              <span className="truncate">{upload.current}</span>
              <span>{Math.round((upload.done / upload.total) * 100)}%</span>
            </div>
            <div className="bar"><i style={{ width: `${(upload.done / upload.total) * 100}%` }} /></div>
          </div>
        )}

        {(busy || notice) && (
          <p className="status">
            {busy && <span className="busy mono">{busy}</span>}
            {notice && <span>{notice}</span>}
          </p>
        )}

        {tab === "search" ? (
          <main>
            <form className="q" onSubmit={runSearch}>
              <input
                ref={searchRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask across your documents…  ( / to focus )"
              />
              <button type="submit" disabled={!query.trim() || !!busy}>Search</button>
            </form>

            <div className="controls">
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
            </div>

            {searched && (
              <p className="resultmeta mono">
                {hits.length} RESULT{hits.length === 1 ? "" : "S"} FOR “{searched}”
              </p>
            )}

            <ol className="hits">
              {hits.map((h, i) => (
                <li key={h.chunk_id} className="card">
                  <div className="hithead">
                    <span className="rank mono">#{i + 1}</span>
                    <span className="file">{h.filename}</span>
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
            {!hits.length && !busy && (
              <p className="empty">No results yet — ingest files above, then search. Scores and match details appear on each card.</p>
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
                    <td className="file">{d.filename}<span className="docid mono">{d.doc_id}</span></td>
                    <td>
                      <span className={`pill mono ${d.status}`}>{d.status.toUpperCase()}</span>
                      {d.error && <span className="err"> — {d.error}</span>}
                    </td>
                    <td className="mono">{d.chunk_count}</td>
                    <td className="acts">
                      <button onClick={() => doReingest(d.doc_id)}>Re-ingest</button>
                      <button className="danger" onClick={() => doDelete(d.doc_id)}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!docs.length && <p className="empty">Library is empty — drop files above.</p>}
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
                <span>Ask across the library — hybrid BM25 + dense retrieval, reranked by answerability.</span>
              </div>
              <div className="mcptool">
                <code className="mono">procure_add_text</code>
                <span>Save transcripts, outputs, notes — chunked, embedded, searchable immediately.</span>
              </div>
              <div className="mcptool">
                <code className="mono">procure_list_documents</code>
                <span>See what is stored — every document with status and chunk counts.</span>
              </div>
              <div className="mcptool">
                <code className="mono">procure://documents/{"{doc_id}"}</code>
                <span>Fetch the full text of a document found via search or listing.</span>
              </div>
              <div className="mcptool">
                <code className="mono">procure://guide</code>
                <span>Agent usage guide — how to search, read documents, and store memories.</span>
              </div>
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
