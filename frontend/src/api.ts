export interface Doc {
  doc_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  error: string;
  created_at: string;
  tags: string[];
  source: string;
  source_uri: string;
  content_hash: string;
  doc_type: string;
}

export interface TagCount {
  tag: string;
  count: number;
}

export interface Job {
  job_id: string;
  kind: string;
  label: string;
  status: string;
  done: number;
  total: number;
  current: string;
  result: unknown;
  error: string;
}

export interface WatchFolder {
  id: number;
  path: string;
  recursive: number;
  created_at: string;
  last_sync: string;
  last_error: string;
  file_count: number;
}

export interface Hit {
  chunk_id: string;
  doc_id: string;
  filename: string;
  text: string;
  doc_type?: string;
  score: number;
  sparse_score?: number;
  dense_score?: number;
  fused_rank?: number;
  answerability?: number;
  entailment?: number;
}

export interface DenseStatus {
  stored_dim: number | null;
  embed_dim: number | null;
  dense_ok: boolean;
  warning: string;
}

export interface Health {
  status: string;
  version: string;
  embeddings: string;
  vectordb: string;
  embed_dim: number;
  chunks: number;
  documents: number;
  search: string;
  rerank: string;
  rerank_backend?: string;
  rerank_models_loaded?: { cross_encoder: boolean; nli: boolean };
  mcp_autostart?: boolean;
  dense?: DenseStatus;
  ocr?: string;
  formats?: string[];
}

/** Shared fetch: same method/parse/error shape for every endpoint, and the
 * server's JSON detail (e.g. conversion guidance) reaches the user instead
 * of a bare status code. */
async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    let detail = "";
    try {
      const j = await r.json();
      detail = typeof j?.detail === "string" ? j.detail : JSON.stringify(j);
    } catch {
      detail = await r.text().catch(() => "");
    }
    throw new Error(detail || `request failed: ${r.status}`);
  }
  return (await r.json()) as T;
}

function json(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export interface DocList {
  documents: Doc[];
  total: number;
}

export interface ListOpts {
  docType?: string;
  source?: string;
  query?: string;
  limit?: number;
  offset?: number;
  sort?: string;
}

export async function listDocs(opts?: ListOpts): Promise<DocList> {
  const p = new URLSearchParams();
  if (opts?.docType) p.set("doc_type", opts.docType);
  if (opts?.source) p.set("source", opts.source);
  if (opts?.query) p.set("q", opts.query);
  if (opts?.limit !== undefined) p.set("limit", String(opts.limit));
  if (opts?.offset) p.set("offset", String(opts.offset));
  if (opts?.sort) p.set("sort", opts.sort);
  const q = p.toString();
  return req<DocList>(`/api/documents${q ? `?${q}` : ""}`);
}

export async function deleteDoc(docId: string): Promise<void> {
  await req(`/api/documents/${docId}`, { method: "DELETE" });
}

export async function reingestDoc(docId: string, background = false): Promise<Doc | Job> {
  if (!background) return req<Doc>(`/api/documents/${docId}/reingest`, { method: "POST" });
  return req<{ job: Job }>(`/api/documents/${docId}/reingest?async=1`, { method: "POST" }).then((j) => j.job);
}

export async function patchDoc(
  docId: string,
  patch: { filename?: string; doc_type?: string }
): Promise<Doc> {
  return req<Doc>(`/api/documents/${docId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
}

export interface SearchScope {
  docIds?: string[];
  tags?: string[];
  source?: string;
  docType?: string;
  since?: string;
}

export async function search(q: string, topK: number, scope?: SearchScope): Promise<Hit[]> {
  const p = new URLSearchParams({ q, top_k: String(topK) });
  for (const d of scope?.docIds ?? []) p.append("doc_id", d);
  for (const t of scope?.tags ?? []) p.append("tag", t);
  if (scope?.source) p.set("source", scope.source);
  if (scope?.docType) p.set("doc_type", scope.docType);
  if (scope?.since) p.set("since", scope.since);
  return req<{ results: Hit[] }>(`/api/search?${p}`).then((j) => j.results);
}

export async function uploadFilesAsync(files: File[], docType?: string): Promise<Job> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const q = docType ? `?async=1&doc_type=${encodeURIComponent(docType)}` : "?async=1";
  return req<{ job: Job }>(`/api/documents/upload${q}`, { method: "POST", body: fd }).then((j) => j.job);
}

export async function getJob(jobId: string): Promise<Job> {
  return req<Job>(`/api/jobs/${jobId}`);
}

export async function cancelJob(jobId: string): Promise<Job> {
  return req<Job>(`/api/jobs/${jobId}`, { method: "DELETE" });
}

export async function addUrl(url: string, tags: string[], docType?: string): Promise<Doc> {
  return req<Doc>(
    "/api/documents/url",
    json({ url, tags, doc_type: docType || "document" })
  );
}

export async function exportLibrary(): Promise<Blob> {
  const r = await fetch("/api/documents/export");
  if (!r.ok) throw new Error(`export failed: ${r.status}`);
  return await r.blob();
}

export async function importLibrary(file: File): Promise<Job> {
  const fd = new FormData();
  fd.append("file", file);
  return req<{ job: Job }>("/api/documents/import", { method: "POST", body: fd }).then((j) => j.job);
}

export async function setDocTags(docId: string, tags: string[]): Promise<{ doc_id: string; tags: string[] }> {
  return req(`/api/documents/${docId}/tags`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tags }),
  });
}

export async function listTags(): Promise<TagCount[]> {
  return req<{ tags: TagCount[] }>("/api/tags").then((j) => j.tags);
}

export async function listWatches(): Promise<{ folders: WatchFolder[]; last_scan: string | null }> {
  return req("/api/watch");
}

export async function addWatch(path: string, recursive: boolean): Promise<WatchFolder> {
  try {
    return await req<WatchFolder>("/api/watch", json({ path, recursive }));
  } catch (e) {
    throw new Error(`${String(e)} — is it a directory on the server?`);
  }
}

export async function removeWatch(id: number, deleteDocs = false): Promise<void> {
  await req(`/api/watch/${id}${deleteDocs ? "?delete_docs=true" : ""}`, { method: "DELETE" });
}

export async function syncWatches(): Promise<Job> {
  return req<{ job: Job }>("/api/watch/sync", { method: "POST" }).then((j) => j.job);
}

export interface SettingsState {
  search: string;
  rerank: string;
  mcp_autostart: boolean;
}

// Backend truthy parsing lives server-side; the UI only needs the two states.
export function rerankIsOff(rerank: string | undefined): boolean {
  return (rerank ?? "on").toLowerCase() !== "on";
}

export async function updateSettings(
  patch: Partial<SettingsState>
): Promise<SettingsState> {
  return req<SettingsState>("/api/settings", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
}

export async function health(): Promise<Health> {
  return req<Health>("/api/health");
}

export interface McpStatus {
  running: boolean;
  pid: number | null;
  url: string;
  uptime_s: number;
  exit_code: number | null;
}

export interface McpLogLine {
  seq: number;
  text: string;
}

export async function mcpStatus(): Promise<McpStatus> {
  return req<McpStatus>("/api/mcp/status");
}

export async function mcpStart(): Promise<McpStatus> {
  return req<McpStatus>("/api/mcp/start", { method: "POST" });
}

export async function mcpStop(): Promise<McpStatus> {
  return req<McpStatus>("/api/mcp/stop", { method: "POST" });
}

export async function mcpLogs(since: number): Promise<{ lines: McpLogLine[]; next: number }> {
  return req(`/api/mcp/logs?since=${since}`);
}

export async function workspaceLogs(since: number): Promise<{ lines: McpLogLine[]; next: number }> {
  return req(`/api/logs?since=${since}`);
}

export interface McpTool {
  name: string;
  description: string;
}

export async function mcpTools(): Promise<McpTool[]> {
  return req<{ tools: McpTool[] }>("/api/mcp/tools").then((j) => j.tools);
}

export interface SkillTargetState {
  id: string;
  label: string;
  path: string;
  detected: boolean;
  installed: boolean;
  installed_version: string;
  bundled_version: string;
  outdated: boolean;
}

export interface SkillStatus {
  skill: string;
  source_found: boolean;
  source_path: string;
  version: string;
  targets: SkillTargetState[];
}

export interface SkillInstallResult {
  id: string;
  path: string;
  action: string;
  detail: string;
}

export async function skillsStatus(): Promise<SkillStatus> {
  return req<SkillStatus>("/api/skills/status");
}

export async function installSkills(force: boolean): Promise<SkillInstallResult[]> {
  return req<{ results: SkillInstallResult[] }>("/api/skills/install", json({ force })).then(
    (j) => j.results
  );
}
