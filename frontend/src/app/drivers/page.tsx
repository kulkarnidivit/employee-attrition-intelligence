"use client";

import { useEffect, useState } from "react";
import { api, GlobalDriver } from "@/lib/api";
import { Bar, Card, ErrorBox } from "@/components/ui";

export default function Drivers() {
  const [data, setData] = useState<GlobalDriver[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.drivers().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <ErrorBox message={error} />;
  if (!data) return <p>Loading...</p>;
  const max = Math.max(...data.map((d) => d.importance));

  return (
    <>
      <h1 className="text-xl font-semibold">Attrition drivers</h1>
      <Card title="What the model's score depends on most">
        <div className="space-y-2">
          {data.map((d) => (
            <div key={d.feature} className="grid grid-cols-[14rem_1fr_4rem] items-center gap-3 text-sm">
              <span className="truncate font-mono text-xs">{d.feature}</span>
              <Bar value={d.importance} max={max} />
              <span className="text-right">{d.importance.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </Card>
      <p className="text-xs text-slate-500">
        Average size of each factor&apos;s effect on the model&apos;s score. These describe the model, not causes.
      </p>
    </>
  );
}
