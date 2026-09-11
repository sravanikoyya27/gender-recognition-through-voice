"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import LoadingSpinner from "@/components/LoadingSpinner";
import { useAnalytics } from "@/hooks/useAnalytics";
import { formatPercent } from "@/lib/utils";

export default function AnalyticsPage() {
  const { summary, loading, error } = useAnalytics();

  const chartData = [
    { gender: "Male", count: summary.byGender.male, fill: "var(--color-male)" },
    { gender: "Female", count: summary.byGender.female, fill: "var(--color-female)" },
  ];

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-h1">Analytics</h1>
        <p className="text-sm text-muted-foreground">
          Aggregate view of stored predictions. Populated once prediction history is persisted.
        </p>
      </header>

      {loading ? <LoadingSpinner label="Loading prediction history..." /> : null}

      {!loading && error ? (
        <p role="status" className="card text-sm text-muted-foreground">
          History is not available yet{error.status ? ` (HTTP ${error.status})` : ""}: {error.message}
        </p>
      ) : null}

      {!loading && !error ? (
        <>
          <div className="grid gap-4 sm:grid-cols-3">
            <Stat label="Total predictions" value={String(summary.total)} />
            <Stat label="Male / Female" value={`${summary.byGender.male} / ${summary.byGender.female}`} />
            <Stat label="Average confidence" value={formatPercent(summary.averageConfidence)} />
          </div>

          {summary.total > 0 ? (
            <div className="card h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="gender" stroke="var(--muted-foreground)" />
                  <YAxis allowDecimals={false} stroke="var(--muted-foreground)" />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="card text-sm text-muted-foreground">No predictions recorded yet.</p>
          )}
        </>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="card">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{value}</p>
    </div>
  );
}
