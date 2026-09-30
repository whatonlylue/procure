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
}

export interface Hit {
  chunk_id: string;
  doc_id: string;
  filename: string;
  text: string;
  score: number;
  sparse_score?: number;
  dense_score?: number;
  fused_rank?: number;
  answerability?: number;
  entailment?: number;
}

export interface Health {
  status: string;
  embeddings: string;
  vectordb: string;
  embed_dim: number;
  chunks: number;
  documents: number;
  search: string;
  rerank: string;
  nli_available?: boolean;
  alpha_nli?: number;
  ocr?: string;
  formats?: string[];
}

export async function uploadFiles(files: File[]): Promise<Doc[]> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const r = await fetch("/api/documents/upload", { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed: ${r.status}`);
  const j = await r.json();
  return j.documents;
}

export async function listDocs(): Promise<Doc[]> {
  const r = await fetch("/api/documents");
  if (!r.ok) throw new Error(`list failed: ${r.status}`);
  return (await r.json()).documents;
}

export async function deleteDoc(docId: string): Promise<void> {
  const r = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`delete failed: ${r.status}`);
}

export async function reingestDoc(docId: string): Promise<Doc> {
  const r = await fetch(`/api/documents/${docId}/reingest`, { method: "POST" });
  if (!r.ok) throw new Error(`reingest failed: ${r.status}`);
  return await r.json();
}

export interface SearchScope {
  docIds?: string[];
  tags?: string[];
  source?: string;
}

export async function search(q: string, topK: number, scope?: SearchScope): Promise<Hit[]> {
  const p = new URLSearchParams({ q, top_k: String(topK) });
  for (const d of scope?.docIds ?? []) p.append("doc_id", d);
  for (const t of scope?.tags ?? []) p.append("tag", t);
  if (scope?.source) p.set("source", scope.source);
  const r = await fetch(`/api/search?${p}`);
  if (!r.ok) throw new Error(`search failed: ${r.status}`);
  return (await r.json()).results;
}

export async function uploadFilesAsync(files: File[]): Promise<Job> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const r = await fetch("/api/documents/upload?async=1", { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed: ${r.status}`);
  return (await r.json()).job;
}

export async function getJob(jobId: string): Promise<Job> {
  const r = await fetch(`/api/jobs/${jobId}`);
  if (!r.ok) throw new Error(`job failed: ${r.status}`);
  return await r.json();
}

export async function cancelJob(jobId: string): Promise<Job> {
  const r = await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`cancel failed: ${r.status}`);
  return await r.json();
}

export async function addUrl(url: string, tags: string[]): Promise<Doc> {
  const r = await fetch("/api/documents/url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, tags }),
  });
  if (!r.ok) throw new Error(`add URL failed: ${r.status}`);
  return await r.json();
}

export async function setDocTags(docId: string, tags: string[]): Promise<{ doc_id: string; tags: string[] }> {
  const r = await fetch(`/api/documents/${docId}/tags`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tags }),
  });
  if (!r.ok) throw new Error(`tags failed: ${r.status}`);
  return await r.json();
}

export async function listTags(): Promise<TagCount[]> {
  const r = await fetch("/api/tags");
  if (!r.ok) throw new Error(`tags failed: ${r.status}`);
  return (await r.json()).tags;
}

export async function listWatches(): Promise<{ folders: WatchFolder[]; last_scan: string | null }> {
  const r = await fetch("/api/watch");
  if (!r.ok) throw new Error(`watch list failed: ${r.status}`);
  return await r.json();
}

export async function addWatch(path: string, recursive: boolean): Promise<WatchFolder> {
  const r = await fetch("/api/watch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, recursive }),
  });
  if (!r.ok) throw new Error(`add watch failed: ${r.status} — is it a directory on the server?`);
  return await r.json();
}

export async function removeWatch(id: number): Promise<void> {
  const r = await fetch(`/api/watch/${id}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`remove watch failed: ${r.status}`);
}

export async function syncWatches(): Promise<Job> {
  const r = await fetch("/api/watch/sync", { method: "POST" });
  if (!r.ok) throw new Error(`sync failed: ${r.status}`);
  return (await r.json()).job;
}

export interface SettingsState {
  search: string;
  rerank: string;
}

export async function updateSettings(
  patch: Partial<SettingsState>
): Promise<SettingsState> {
  const r = await fetch("/api/settings", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!r.ok) throw new Error(`settings failed: ${r.status}`);
  return await r.json();
}

export async function health(): Promise<Health> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`health failed: ${r.status}`);
  return await r.json();
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
  const r = await fetch("/api/mcp/status");
  if (!r.ok) throw new Error(`mcp status failed: ${r.status}`);
  return await r.json();
}

export async function mcpStart(): Promise<McpStatus> {
  const r = await fetch("/api/mcp/start", { method: "POST" });
  if (!r.ok) throw new Error(`mcp start failed: ${r.status}`);
  return await r.json();
}

export async function mcpStop(): Promise<McpStatus> {
  const r = await fetch("/api/mcp/stop", { method: "POST" });
  if (!r.ok) throw new Error(`mcp stop failed: ${r.status}`);
  return await r.json();
}

export async function mcpLogs(since: number): Promise<{ lines: McpLogLine[]; next: number }> {
  const r = await fetch(`/api/mcp/logs?since=${since}`);
  if (!r.ok) throw new Error(`mcp logs failed: ${r.status}`);
  return await r.json();
}
