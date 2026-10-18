export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type Scalar = string | number;
export type Record27 = Record<string, Scalar>;

export type Summary = {
  employees: number;
  actual_attrition_rate: number;
  flagged: number;
  by_risk_level: { risk_level: string; employees: number; actual_attrition_rate: number }[];
  note: string;
};
export type GlobalDriver = { feature: string; importance: number };
export type EmployeeRow = {
  EmployeeNumber: number;
  Department: string;
  JobRole: string;
  JobLevel: number;
  YearsAtCompany: number;
  OverTime: string;
  in_training: boolean;
  probability: number;
  risk_level: string;
  flagged: boolean;
  left: boolean;
};
export type EmployeeList = { total: number; items: EmployeeRow[] };
export type Driver = { feature: string; contribution: number; direction: string };
export type EmployeeDetail = {
  employee_number: number;
  in_training: boolean;
  profile: Record27;
  score: { probability: number; risk_level: string; flagged: boolean; cutoff: number };
  drivers: Driver[];
  review_areas: string[];
  record: Record27;
};
export type SimulateResult = {
  before: { probability: number; risk_level: string };
  after: { probability: number; risk_level: string };
  change_in_probability: number;
  note: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {};
  if (init?.method === "POST") headers["Content-Type"] = "application/json";
  const key = process.env.NEXT_PUBLIC_API_KEY;
  if (key) headers["X-API-Key"] = key;
  const res = await fetch(`${API_URL}${path}`, { ...init, headers, cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return (await res.json()) as T;
}

export const api = {
  summary: () => request<Summary>("/summary"),
  drivers: () => request<GlobalDriver[]>("/drivers/global"),
  employees: (level: string, limit: number, offset: number) =>
    request<EmployeeList>(
      `/employees?limit=${limit}&offset=${offset}${level !== "All" ? `&risk_level=${level}` : ""}`,
    ),
  employee: (id: string) => request<EmployeeDetail>(`/employees/${id}`),
  simulate: (employee: Record27, changes: Record27) =>
    request<SimulateResult>("/simulate", { method: "POST", body: JSON.stringify({ employee, changes }) }),
};
