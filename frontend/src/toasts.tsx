import { useEffect } from "react";

export type ToastKind = "error" | "success" | "info";

export interface Toast {
  id: number;
  kind: ToastKind;
  text: string;
}

const DISMISS_MS: Record<ToastKind, number> = {
  error: 8000,
  success: 6000,
  info: 5000,
};

function ToastItem({
  toast,
  onDismiss,
}: {
  toast: Toast;
  onDismiss: (id: number) => void;
}) {
  useEffect(() => {
    const ms = DISMISS_MS[toast.kind] ?? 5000;
    const t = window.setTimeout(() => onDismiss(toast.id), ms);
    return () => window.clearTimeout(t);
  }, [toast.id, toast.kind, onDismiss]);

  return (
    <div
      className={`toast mono ${toast.kind}`}
      role={toast.kind === "error" ? "alert" : "status"}
    >
      <span className="toast-text">{toast.text}</span>
      <button
        type="button"
        className="toast-close"
        onClick={() => onDismiss(toast.id)}
        aria-label="Dismiss notification"
        title="Dismiss"
      >
        ✕
      </button>
    </div>
  );
}

export function ToastStack({
  toasts,
  onDismiss,
}: {
  toasts: Toast[];
  onDismiss: (id: number) => void;
}) {
  if (!toasts.length) return null;
  return (
    <div className="toasts" role="region" aria-label="Notifications">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onDismiss={onDismiss} />
      ))}
    </div>
  );
}
