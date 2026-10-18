"use client";

import { useEffect, useState } from "react";
import { api, Summary } from "@/lib/api";
import { Bar, Card, ErrorBox, RiskBadge, pct } from "@/components/ui";

export default function Overview() {
  const [data, setData] = useState<Summary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.summary().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <ErrorBox message={error} />;
  if (!data) return <p>Loading...</p>;

  return (
    <>
      <h1 className="text-xl font-semibold">Executive overview</h1>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card title="Employees"><p className="text-3xl font-semibold">{data.employees}</p></Card>
        <Card title="Actual attrition rate">
          <p className="text-3xl font-semibold">{pct(data.actual_attrition_rate)}</p>
        </Card>
        <Card title="Flagged by the model"><p className="text-3xl font-semibold">{data.flagged}</p></Card>
      </div>
      <Card title="Actual attrition by risk level">
        <div className="space-y-3">
          {data.by_risk_level.map((row) => (
            <div key={row.risk_level} className="grid grid-cols-[6rem_1fr_9rem] items-center gap-3 text-sm">
              <RiskBadge level={row.risk_level} />
              <Bar value={row.actual_attrition_rate} max={1} />
              <span>{pct(row.actual_attrition_rate)} of {row.employees}</span>
            </div>
          ))}
        </div>
      </Card>
      <p className="text-xs text-slate-500">{data.note}</p>
    </>
  );
}
