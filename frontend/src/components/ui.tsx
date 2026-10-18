export const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export function RiskBadge({ level }: { level: string }) {
  const colors: Record<string, string> = {
    High: "bg-red-100 text-red-800",
    Medium: "bg-amber-100 text-amber-800",
    Low: "bg-green-100 text-green-800",
  };
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${colors[level] ?? ""}`}>{level}</span>;
}

export function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">{title}</h2>
      {children}
    </section>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-800">
      Could not reach the API. Is it running (uvicorn api.main:app --reload)? <br />
      <span className="font-mono text-xs">{message}</span>
    </div>
  );
}

export function Bar({ value, max }: { value: number; max: number }) {
  return (
    <div className="h-3 w-full rounded bg-slate-100">
      <div className="h-3 rounded bg-indigo-500" style={{ width: `${Math.min(100, (value / max) * 100)}%` }} />
    </div>
  );
}
