"use client";

import { useParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { api, EmployeeDetail, Record27, Scalar, SimulateResult } from "@/lib/api";
import { Card, ErrorBox, RiskBadge, pct } from "@/components/ui";

const CONTROLS: { field: string; options: Scalar[] }[] = [
  { field: "OverTime", options: ["Yes", "No"] },
  { field: "BusinessTravel", options: ["Non-Travel", "Travel_Rarely", "Travel_Frequently"] },
  { field: "StockOptionLevel", options: [0, 1, 2, 3] },
  { field: "JobSatisfaction", options: [1, 2, 3, 4] },
  { field: "EnvironmentSatisfaction", options: [1, 2, 3, 4] },
  { field: "WorkLifeBalance", options: [1, 2, 3, 4] },
];

function EmployeeRoute() {
  const { id } = useParams<{ id: string }>();
  return <Employee360 key={id} id={id} />;
}

export default function EmployeePage() {
  return (
    <Suspense fallback={<p>Loading...</p>}>
      <EmployeeRoute />
    </Suspense>
  );
}

function Employee360({ id }: { id: string }) {
  const [detail, setDetail] = useState<EmployeeDetail | null>(null);
  const [scenario, setScenario] = useState<Record27>({});
  const [result, setResult] = useState<SimulateResult | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.employee(id).then(setDetail).catch((e) => setError(String(e)));
  }, [id]);

  const changes = useMemo(() => {
    const out: Record27 = {};
    if (!detail) return out;
    for (const c of CONTROLS) {
      const v = scenario[c.field];
      if (v !== undefined && v !== detail.record[c.field]) out[c.field] = v;
    }
    return out;
  }, [scenario, detail]);
  const hasChanges = Object.keys(changes).length > 0;

  useEffect(() => {
    if (!detail || !hasChanges) return;
    api.simulate(detail.record, changes).then(setResult).catch((e) => setError(String(e)));
  }, [detail, changes, hasChanges]);

  if (error) return <ErrorBox message={error} />;
  if (!detail) return <p>Loading...</p>;

  return (
    <>
      <h1 className="text-xl font-semibold">Employee #{detail.employee_number}</h1>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Card title="Profile">
          <dl className="grid grid-cols-2 gap-1 text-sm">
            {Object.entries(detail.profile).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-slate-500">{k}</dt>
                <dd>{String(v)}</dd>
              </div>
            ))}
          </dl>
        </Card>
        <Card title="Model output">
          <p className="text-3xl font-semibold">{pct(detail.score.probability)}</p>
          <p className="mt-1 text-sm">Risk level: <RiskBadge level={detail.score.risk_level} /></p>
          <p className="mt-1 text-xs text-slate-500">
            Flagged above the {pct(detail.score.cutoff)} cutoff: {detail.score.flagged ? "yes" : "no"}
          </p>
          {detail.in_training && (
            <p className="mt-2 text-xs text-amber-700">The model trained on this employee, so this score is optimistic.</p>
          )}
        </Card>
      </div>

      <Card title="Factors that matter most to this score (not causes)">
        <ul className="space-y-1 text-sm">
          {detail.drivers.map((d) => (
            <li key={d.feature}>
              <span className="font-mono text-xs">{d.feature}</span>{" "}
              <span className={d.direction === "raises" ? "text-red-700" : "text-green-700"}>
                {d.direction} the score
              </span>{" "}
              <span className="text-slate-500">({d.contribution.toFixed(2)})</span>
            </li>
          ))}
        </ul>
        {detail.review_areas.length > 0 && (
          <div className="mt-3 text-sm">
            <p className="font-medium">Possible areas for HR review (decision support, not a decision):</p>
            <ul className="list-disc pl-5">
              {detail.review_areas.map((a) => <li key={a}>{a}</li>)}
            </ul>
          </div>
        )}
      </Card>

      <Card title="What-if scenario">
        <p className="mb-3 text-xs text-slate-500">
          A model scenario: it shows how the model&apos;s output changes, not that changing these factors would change real outcomes.
        </p>
        <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
          {CONTROLS.map((c) => (
            <label key={c.field} className="flex flex-col gap-1">
              {c.field}
              <select
                className="rounded border p-1"
                value={String(scenario[c.field] ?? detail.record[c.field])}
                onChange={(e) =>
                  setScenario({
                    ...scenario,
                    [c.field]: typeof c.options[0] === "number" ? Number(e.target.value) : e.target.value,
                  })
                }
              >
                {c.options.map((o) => <option key={String(o)} value={String(o)}>{String(o)}</option>)}
              </select>
            </label>
          ))}
        </div>
        {hasChanges && result ? (
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div>Before: <b>{pct(result.before.probability)}</b> <RiskBadge level={result.before.risk_level} /></div>
            <div>
              After: <b>{pct(result.after.probability)}</b> <RiskBadge level={result.after.risk_level} />{" "}
              <span className="text-slate-500">({(result.change_in_probability * 100).toFixed(1)} points)</span>
            </div>
          </div>
        ) : (
          <p className="mt-4 text-sm text-slate-500">Change a control to compare the model&apos;s output.</p>
        )}
      </Card>
    </>
  );
}
