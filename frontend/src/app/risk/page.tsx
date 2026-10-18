"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, EmployeeList } from "@/lib/api";
import { Card, ErrorBox, RiskBadge, pct } from "@/components/ui";

const LIMIT = 25;
const LEVELS = ["All", "High", "Medium", "Low"];

export default function RiskCenter() {
  const [level, setLevel] = useState("High");
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<EmployeeList | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.employees(level, LIMIT, offset).then(setData).catch((e) => setError(String(e)));
  }, [level, offset]);

  if (error) return <ErrorBox message={error} />;

  return (
    <>
      <h1 className="text-xl font-semibold">Risk center</h1>
      <div className="flex gap-2 text-sm">
        {LEVELS.map((l) => (
          <button
            key={l}
            onClick={() => { setLevel(l); setOffset(0); }}
            className={`rounded border px-3 py-1 ${level === l ? "bg-slate-900 text-white" : "bg-white"}`}
          >
            {l}
          </button>
        ))}
      </div>
      <Card title={data ? `${data.total} employees` : "Loading..."}>
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase text-slate-500">
            <tr>
              <th className="py-1">Employee</th><th>Role</th><th>Overtime</th><th>Probability</th><th>Risk</th><th>Left?</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map((e) => (
              <tr key={e.EmployeeNumber} className="border-t border-slate-100">
                <td className="py-1">
                  <Link className="text-indigo-600 hover:underline" href={`/employee/${e.EmployeeNumber}`}>
                    #{e.EmployeeNumber}
                  </Link>
                  {e.in_training && <span className="ml-1 text-xs text-slate-400" title="The model trained on this employee">*</span>}
                </td>
                <td>{e.JobRole}</td>
                <td>{e.OverTime}</td>
                <td>{pct(e.probability)}</td>
                <td><RiskBadge level={e.risk_level} /></td>
                <td>{e.left ? "Yes" : "No"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-3 flex items-center gap-3 text-sm">
          <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - LIMIT))} className="rounded border px-3 py-1 disabled:opacity-40">Previous</button>
          <button disabled={!data || offset + LIMIT >= data.total} onClick={() => setOffset(offset + LIMIT)} className="rounded border px-3 py-1 disabled:opacity-40">Next</button>
        </div>
      </Card>
      <p className="text-xs text-slate-500">* the model trained on this employee, so their score looks better than it would for a new person.</p>
    </>
  );
}
