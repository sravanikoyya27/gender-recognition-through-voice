"use client";

import { memo, type ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import Skeleton from "@/components/Skeleton";
import { type ConfidenceBucket, modelLabel, type TimelinePoint } from "@/lib/analytics";
import type { ApiError, ModelPerformanceResponse } from "@/types/api";

/* ---------- shared chrome ---------- */

export function ChartCard({
  title,
  description,
  children,
  loading,
  error,
  onRetry,
  height = 288,
}: {
  title: string;
  description?: ReactNode;
  children: ReactNode;
  loading?: boolean;
  error?: ApiError | null;
  onRetry?: () => void;
  height?: number;
}) {
  return (
    <section className="card" aria-labelledby={`chart-${title.replace(/\s+/g, "-").toLowerCase()}`}>
      <h3 id={`chart-${title.replace(/\s+/g, "-").toLowerCase()}`} className="text-h3">
        {title}
      </h3>
      {description ? <p className="mt-1 text-xs text-muted-foreground">{description}</p> : null}
      <div className="mt-4" style={{ minHeight: height }}>
        {error ? (
          <div role="alert" className="flex h-full flex-col items-start gap-2 text-sm">
            <span>
              Could not load{error.status ? ` (HTTP ${error.status})` : ""}: {error.message}
            </span>
            {onRetry ? (
              <button type="button" className="btn-secondary" onClick={onRetry}>
                Retry
              </button>
            ) : null}
          </div>
        ) : loading ? (
          <Skeleton className="w-full" />
        ) : (
          children
        )}
      </div>
    </section>
  );
}

function EmptyChart({ message }: { message: string }) {
  return <p className="flex h-72 items-center justify-center text-sm text-muted-foreground">{message}</p>;
}

const tooltipStyle = {
  contentStyle: { background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 },
  labelStyle: { color: "var(--foreground)" },
  itemStyle: { color: "var(--foreground)" },
};

/* ---------- a) predictions timeline ---------- */

export const TimelineChart = memo(function TimelineChart({ data }: { data: TimelinePoint[] }) {
  const total = data.reduce((sum, p) => sum + p.count, 0);
  if (total === 0) return <EmptyChart message="No predictions in this period." />;
  return (
    <div className="h-72" role="img" aria-label={`Line chart of predictions per day, ${total} predictions over ${data.length} days`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} minTickGap={24} />
          <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} />
          <Tooltip {...tooltipStyle} formatter={(value: number) => [value, "Predictions"]} labelFormatter={(label, payload) => payload?.[0]?.payload?.date ?? label} />
          <Line type="monotone" dataKey="count" name="Predictions" stroke="#7c3aed" strokeWidth={2} dot={data.length <= 45} activeDot={{ r: 5 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});

/* ---------- b) gender distribution ---------- */

export const GenderDonut = memo(function GenderDonut({ male, female }: { male: number; female: number }) {
  const total = male + female;
  if (total === 0) return <EmptyChart message="No predictions in this period." />;
  const pct = (n: number) => `${((100 * n) / total).toFixed(1)}%`;
  const data = [
    { name: "Male", value: male, fill: "var(--color-male)" },
    { name: "Female", value: female, fill: "var(--color-female)" },
  ];
  return (
    <div className="h-72" role="img" aria-label={`Donut chart: male ${male} (${pct(male)}), female ${female} (${pct(female)})`}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" innerRadius="55%" outerRadius="80%" paddingAngle={2} stroke="none" label={({ value }) => `${value} (${pct(Number(value))})`}>
            {data.map((entry) => (
              <Cell key={entry.name} fill={entry.fill} />
            ))}
          </Pie>
          <Tooltip {...tooltipStyle} formatter={(value: number, name: string) => [`${value} (${pct(value)})`, name]} />
          <Legend verticalAlign="bottom" formatter={(value: string) => `${value}: ${value === "Male" ? male : female} (${pct(value === "Male" ? male : female)})`} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
});

/* ---------- c) confidence distribution ---------- */

export const ConfidenceHistogram = memo(function ConfidenceHistogram({
  buckets,
  threshold,
}: {
  buckets: ConfidenceBucket[];
  threshold: number;
}) {
  const total = buckets.reduce((sum, b) => sum + b.count, 0);
  if (total === 0) return <EmptyChart message="No predictions in this period." />;
  const summary = buckets.map((b) => `${b.label}: ${b.count}`).join(", ");
  return (
    <div className="h-72" role="img" aria-label={`Bar chart of prediction confidence buckets. ${summary}`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={buckets} margin={{ top: 16, right: 16, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="label" tick={{ fontSize: 12, fill: "var(--muted-foreground)" }} />
          <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} />
          <Tooltip {...tooltipStyle} cursor={{ fill: "var(--muted)" }} formatter={(value: number) => [value, "Predictions"]} />
          <Bar dataKey="count" name="Predictions" radius={[4, 4, 0, 0]}>
            {buckets.map((bucket) => (
              // Buckets entirely below the threshold are dimmed (the threshold highlights, it does not hide).
              <Cell key={bucket.key} fill={bucket.color} fillOpacity={threshold > 0 && bucket.max <= threshold ? 0.3 : 1} />
            ))}
            <LabelList dataKey="count" position="top" fontSize={12} fill="var(--foreground)" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
});

/* ---------- d) model performance ---------- */

export const ModelPerformanceChart = memo(function ModelPerformanceChart({ data }: { data: ModelPerformanceResponse }) {
  const rows = data.models.map((m) => ({
    key: m.model_key,
    label: modelLabel(m.model_key),
    accuracy: Number((m.accuracy * 100).toFixed(2)),
    current: m.model_key === data.current_model,
    n: m.total_predictions,
  }));
  if (rows.length === 0) return <EmptyChart message="No model metrics available." />;
  const min = Math.min(...rows.map((r) => r.accuracy));
  const floor = Math.max(0, Math.floor((min - 2) / 5) * 5);
  const summary = rows.map((r) => `${r.label} ${r.accuracy}%${r.current ? " (current)" : ""}`).join(", ");
  return (
    <div className="h-72" role="img" aria-label={`Bar chart of test accuracy per model. ${summary}`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 16, right: 16, left: -8, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} interval={0} />
          <YAxis domain={[floor, 100]} unit="%" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} />
          <Tooltip
            {...tooltipStyle}
            cursor={{ fill: "var(--muted)" }}
            formatter={(value: number, _name, item) => [
              `${value}% on ${item.payload.n ?? "?"} test rows${item.payload.current ? " · current model" : ""}`,
              "Test accuracy",
            ]}
          />
          <Bar dataKey="accuracy" name="Test accuracy" radius={[4, 4, 0, 0]}>
            {rows.map((row) => (
              <Cell key={row.key} fill={row.current ? "#7c3aed" : "#a1a1aa"} />
            ))}
            <LabelList dataKey="accuracy" position="top" fontSize={11} fill="var(--foreground)" formatter={(v: number) => `${v}%`} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
});
