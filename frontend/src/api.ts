export interface Doc {
  doc_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  error: string;
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

export async function search(q: string, topK: number): Promise<Hit[]> {
  const p = new URLSearchParams({ q, top_k: String(topK) });
  const r = await fetch(`/api/search?${p}`);
  if (!r.ok) throw new Error(`search failed: ${r.status}`);
  return (await r.json()).results;
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
