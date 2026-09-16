"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import {
  AlertCircle,
  ArrowDownAZ,
  ArrowDownWideNarrow,
  Bookmark,
  BookmarkCheck,
  ChevronDown,
  ChevronUp,
  Download,
  Link2,
  RotateCcw,
  Search,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import LoadingSpinner from "@/components/LoadingSpinner";
import { useCountUp } from "@/hooks/useCountUp";
import { useModelFeatures } from "@/hooks/useModelFeatures";
import { cn, formatPercent } from "@/lib/utils";
import type { ApiError, ModelFeaturesResponse } from "@/types/api";
import { FEATURE_NAMES, type Prediction, type PredictionStatus } from "@/types/models";

export interface ResultsDisplayProps {
  status: PredictionStatus;
  result: Prediction | null;
  error: ApiError | null;
  /** Reset the recorder / upload form and clear the result. */
  onTryAnother?: () => void;
  /** Download the analysed audio (only shown when the audio is still available). */
  onDownload?: () => void;
  /** Share: defaults to copying a link to this prediction. */
  onShare?: (prediction: Prediction) => void;
  /** Save: defaults to reporting the server-side persistence state. */
  onSave?: (prediction: Prediction) => void;
  className?: string;
}

const MODEL_LABELS: Record<string, string> = {
  MLP: "Neural Network (MLP)",
  RandomForest: "Random Forest",
  SVM: "Support Vector Machine",
  GradientBoosting: "Gradient Boosting",
};

const FEATURE_DESCRIPTIONS: Record<string, string> = {
  meanfreq: "mean frequency (kHz)",
  sd: "frequency standard deviation",
  median: "median frequency (kHz)",
  Q25: "first quartile (kHz)",
  Q75: "third quartile (kHz)",
  IQR: "interquartile range (kHz)",
  skew: "spectral skewness",
  kurt: "spectral kurtosis",
  "sp.ent": "spectral entropy",
  sfm: "spectral flatness",
  mode: "mode frequency (kHz)",
  centroid: "frequency centroid (kHz)",
  meanfun: "mean fundamental frequency (kHz)",
  minfun: "minimum fundamental frequency (kHz)",
  maxfun: "maximum fundamental frequency (kHz)",
  meandom: "mean dominant frequency (kHz)",
  mindom: "minimum dominant frequency (kHz)",
  maxdom: "maximum dominant frequency (kHz)",
  dfrange: "dominant frequency range (kHz)",
  modindx: "modulation index",
};

export default function ResultsDisplay({
  status,
  result,
  error,
  onTryAnother,
  onDownload,
  onShare,
  onSave,
  className,
}: ResultsDisplayProps) {
  return (
    <section aria-live="polite" aria-label="Prediction results" className={cn("flex flex-col gap-4", className)}>
      {status === "idle" && !result ? (
        <div className="card">
          <h2 className="text-h3">Result</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Record or upload a voice sample to see the prediction here.
          </p>
        </div>
      ) : null}

      {status === "uploading" || status === "analyzing" ? (
        <div className="card flex items-center gap-4">
          <LoadingSpinner
            size="lg"
            label={status === "uploading" ? "Uploading audio…" : "Extracting features and running the neural network…"}
          />
        </div>
      ) : null}

      {status === "error" && error ? (
        <div role="alert" className="card border-red-200 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          <p className="flex items-center gap-2 font-medium">
            <AlertCircle className="h-4 w-4 shrink-0" aria-hidden />
            Prediction failed{error.status ? ` (HTTP ${error.status})` : ""}
          </p>
          <p className="mt-1 text-sm">{error.message}</p>
          {onTryAnother ? (
            <button type="button" onClick={onTryAnother} className="btn-secondary mt-3">
              <RotateCcw className="h-4 w-4" aria-hidden />
              Try another recording
            </button>
          ) : null}
        </div>
      ) : null}

      {status === "success" && result ? (
        <ResultPanel
          key={result.prediction_id ?? result.created_at ?? "result"}
          result={result}
          onTryAnother={onTryAnother}
          onDownload={onDownload}
          onShare={onShare}
          onSave={onSave}
        />
      ) : null}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Result panel
// ---------------------------------------------------------------------------

function ResultPanel({
  result,
  onTryAnother,
  onDownload,
  onShare,
  onSave,
}: Omit<ResultsDisplayProps, "status" | "error" | "className"> & { result: Prediction }) {
  const isMale = result.predicted_gender === "male";
  const label = isMale ? "Male" : "Female";
  const glyph = isMale ? "♂" : "♀"; // ♂ / ♀ : a non-colour cue (lucide 0.292 has no gender icons)
  const confidence = result.confidence;
  const animated = useCountUp(confidence, 1200);
  const tone = confidence > 0.9 ? "high" : confidence >= 0.7 ? "medium" : "low";
  const { data: meta } = useModelFeatures();
  const features = useMemo(() => buildFeatureRows(result, meta), [result, meta]);

  return (
    <div className="animate-fade-in-slow grid min-w-0 gap-4 lg:grid-cols-2">
      {/* Left column: prediction */}
      <div className="flex min-w-0 flex-col gap-4">
        <div
          className={cn(
            "card flex flex-col items-center gap-4 py-10 text-center shadow-lg",
            isMale ? "border-male-200 dark:border-male-800" : "border-female-200 dark:border-female-800",
          )}
        >
          <p className="sr-only">
            Predicted gender {label} with {formatPercent(confidence, 2)} confidence.
          </p>
          <div className="animate-scale-in flex flex-col items-center gap-2" aria-hidden>
            <span className={cn("text-6xl font-bold leading-none", isMale ? "text-male" : "text-female")}>
              {glyph}
            </span>
            <p
              className={cn(
                // 100 px+ on tablets and up; on narrow phones the word must still fit the viewport.
                "text-[min(6.5rem,21vw)] font-black uppercase leading-none tracking-tight sm:text-[7.5rem]",
                isMale ? "text-male" : "text-female",
              )}
            >
              {label}
            </p>
          </div>
          <span className={isMale ? "badge-male" : "badge-female"}>{label} voice</span>
        </div>

        <ConfidenceCard confidence={confidence} animated={animated} tone={tone} isMale={isMale} />

        <MetadataCard result={result} />

        <ActionsCard result={result} onTryAnother={onTryAnother} onDownload={onDownload} onShare={onShare} onSave={onSave} />
      </div>

      {/* Right column: features */}
      <div className="flex min-w-0 flex-col gap-4">
        <FeatureCharts rows={features} meta={meta} />
        <FeatureTable rows={features} meta={meta} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Confidence
// ---------------------------------------------------------------------------

type Tone = "high" | "medium" | "low";
const TONE_CLASSES: Record<Tone, { bar: string; text: string; label: string }> = {
  high: { bar: "bg-green-500", text: "text-green-700 dark:text-green-400", label: "High confidence" },
  medium: { bar: "bg-yellow-400", text: "text-yellow-700 dark:text-yellow-300", label: "Moderate confidence" },
  low: { bar: "bg-red-500", text: "text-red-700 dark:text-red-400", label: "Low confidence" },
};
const TONE_STROKE: Record<Tone, string> = { high: "#22c55e", medium: "#facc15", low: "#ef4444" };

function ConfidenceCard({
  confidence,
  animated,
  tone,
  isMale,
}: {
  confidence: number;
  animated: number;
  tone: Tone;
  isMale: boolean;
}) {
  const classes = TONE_CLASSES[tone];
  const radius = 44;
  const circumference = 2 * Math.PI * radius;
  const pctText = formatPercent(animated, 2);

  return (
    <div className="card flex flex-col gap-4 sm:flex-row sm:items-center">
      <svg
        viewBox="0 0 100 100"
        className="h-28 w-28 shrink-0 self-center"
        role="img"
        aria-label={`Confidence gauge ${formatPercent(confidence, 2)}`}
      >
        <circle cx="50" cy="50" r={radius} fill="none" stroke="var(--muted)" strokeWidth="10" />
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke={TONE_STROKE[tone]}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - animated)}
          transform="rotate(-90 50 50)"
        />
        <text x="50" y="54" textAnchor="middle" fontSize="16" fontWeight="700" fill="currentColor">
          {Math.round(animated * 100)}%
        </text>
      </svg>

      <div className="flex flex-1 flex-col gap-2">
        <div className="flex items-baseline justify-between gap-3">
          <p className="text-h3">Confidence</p>
          <p className={cn("font-mono text-3xl font-semibold tabular-nums", classes.text)} aria-hidden>
            {pctText}
          </p>
        </div>
        <div
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(confidence * 100)}
          aria-label="Prediction confidence"
          className="h-3 w-full overflow-hidden rounded-full bg-muted"
        >
          <div className={cn("h-full rounded-full", classes.bar)} style={{ width: `${animated * 100}%` }} />
        </div>
        <p className={cn("text-sm", classes.text)}>
          {classes.label} · P({isMale ? "male" : "female"}) = {formatPercent(confidence, 2)}
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

function MetadataCard({ result }: { result: Prediction }) {
  const recorded = result.created_at ? formatDateTime(result.created_at) : "–";
  const items: [string, string][] = [
    ["Duration", result.duration_seconds != null ? `${result.duration_seconds.toFixed(2)} seconds` : "–"],
    ["Recording time", recorded],
    ["Model used", MODEL_LABELS[result.model_used ?? ""] ?? result.model_used ?? "–"],
    ["Inference time", result.inference_time_ms != null ? `${result.inference_time_ms.toFixed(1)} ms` : "–"],
    ["Sample rate", result.sample_rate ? `${result.sample_rate} Hz` : "–"],
    ["File", result.audio_filename ?? "–"],
  ];
  return (
    <div className="card">
      <h3 className="text-h3">Audio metadata</h3>
      <dl className="mt-3 grid grid-cols-1 gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
        {items.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3 border-b border-border py-1 sm:block sm:border-0">
            <dt className="text-xs uppercase tracking-wide text-muted-foreground">{k}</dt>
            <dd className="truncate font-medium" title={v}>
              {v}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function formatDateTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  // The locale supplies the date/time joiner ("September 11, 2026 at 3:12 PM").
  return new Intl.DateTimeFormat(undefined, { dateStyle: "long", timeStyle: "short" }).format(date);
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

function ActionsCard({
  result,
  onTryAnother,
  onDownload,
  onShare,
  onSave,
}: Omit<ResultsDisplayProps, "status" | "error" | "className"> & { result: Prediction }) {
  const [shareState, setShareState] = useState<"idle" | "copied" | "failed">("idle");
  const saved = result.prediction_id != null;

  async function handleShare() {
    if (onShare) {
      onShare(result);
      return;
    }
    const url = saved
      ? `${window.location.origin}/analytics?prediction=${result.prediction_id}`
      : window.location.href;
    try {
      await navigator.clipboard.writeText(url);
      setShareState("copied");
    } catch {
      setShareState("failed");
    }
    setTimeout(() => setShareState("idle"), 2500);
  }

  return (
    <div className="card">
      <h3 className="text-h3">Actions</h3>
      <div className="mt-3 flex flex-wrap gap-2">
        {onTryAnother ? (
          <button type="button" onClick={onTryAnother} className="btn-primary" aria-label="Try another recording">
            <RotateCcw className="h-4 w-4" aria-hidden />
            Try another recording
          </button>
        ) : null}
        {onDownload ? (
          <button type="button" onClick={onDownload} className="btn-secondary" aria-label="Download audio file">
            <Download className="h-4 w-4" aria-hidden />
            Download audio
          </button>
        ) : null}
        <button
          type="button"
          onClick={() => (onSave ? onSave(result) : undefined)}
          className="btn-secondary"
          aria-label={saved ? `Prediction saved with id ${result.prediction_id}` : "Save prediction"}
          disabled={saved && !onSave}
          title={saved ? "Stored in your history" : "Sign in to keep predictions in your history"}
        >
          {saved ? <BookmarkCheck className="h-4 w-4" aria-hidden /> : <Bookmark className="h-4 w-4" aria-hidden />}
          {saved ? `Saved (#${result.prediction_id})` : "Save prediction"}
        </button>
        <button type="button" onClick={() => void handleShare()} className="btn-secondary" aria-label="Share prediction link">
          <Link2 className="h-4 w-4" aria-hidden />
          {shareState === "copied" ? "Link copied" : shareState === "failed" ? "Copy failed" : "Share link"}
        </button>
        <Link
          href={`/analytics?gender=${result.predicted_gender}`}
          className="btn-secondary"
          aria-label={`View similar ${result.predicted_gender} predictions`}
        >
          <Search className="h-4 w-4" aria-hidden />
          Similar predictions
        </Link>
      </div>
      {!saved ? (
        <p className="mt-2 text-xs text-muted-foreground">Sign in before predicting to keep results in your history.</p>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Features: rows, table, charts
// ---------------------------------------------------------------------------

interface FeatureRow {
  name: string;
  value: number;
  importance: number | null;
  rank: number | null; // 1 = most important
  z: number | null; // standardised against the training split
}

function buildFeatureRows(result: Prediction, meta: ModelFeaturesResponse | null): FeatureRow[] {
  const values = result.features ?? {};
  const importance = meta?.importance?.values ?? null;
  const ranked = importance ? Object.keys(importance) : [];
  return FEATURE_NAMES.map((name) => {
    const value = Number(values[name] ?? NaN);
    const mean = meta?.reference?.mean[name];
    const std = meta?.reference?.std[name];
    const z = mean != null && std ? (value - mean) / std : null;
    const idx = ranked.indexOf(name);
    return {
      name,
      value,
      importance: importance ? (importance[name] ?? 0) : null,
      rank: idx >= 0 ? idx + 1 : null,
      z: z != null && Number.isFinite(z) ? z : null,
    };
  });
}

type SortKey = "importance" | "name" | "value";

function FeatureTable({ rows, meta }: { rows: FeatureRow[]; meta: ModelFeaturesResponse | null }) {
  const [open, setOpen] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>(meta?.importance ? "importance" : "name");
  const sorted = useMemo(() => {
    const copy = [...rows];
    if (sortKey === "name") copy.sort((a, b) => a.name.localeCompare(b.name));
    else if (sortKey === "value") copy.sort((a, b) => b.value - a.value);
    else copy.sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99));
    return copy;
  }, [rows, sortKey]);

  return (
    <div className="card">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls="feature-table"
        className="flex w-full items-center justify-between gap-2 text-left"
      >
        <span className="text-h3">Acoustic features ({rows.length})</span>
        {open ? <ChevronUp className="h-5 w-5" aria-hidden /> : <ChevronDown className="h-5 w-5" aria-hidden />}
      </button>
      {!open ? (
        <p className="mt-1 text-xs text-muted-foreground">
          {meta?.importance
            ? `Top 5 by importance: ${rows
                .filter((r) => r.rank != null && r.rank <= 5)
                .sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0))
                .map((r) => r.name)
                .join(", ")}`
            : "Expand to see all 20 extracted values."}
        </p>
      ) : null}

      {open ? (
        <div id="feature-table" className="mt-3 flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2 text-xs" role="group" aria-label="Sort features">
            <span className="text-muted-foreground">Sort by</span>
            {(["importance", "name", "value"] as SortKey[]).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => setSortKey(key)}
                aria-pressed={sortKey === key}
                disabled={key === "importance" && !meta?.importance}
                className={cn(
                  "inline-flex items-center gap-1 rounded border px-2 py-1",
                  sortKey === key ? "border-brand bg-brand-50 text-brand-700 dark:bg-brand-900/30" : "border-border",
                )}
              >
                {key === "name" ? <ArrowDownAZ className="h-3 w-3" aria-hidden /> : <ArrowDownWideNarrow className="h-3 w-3" aria-hidden />}
                {key}
              </button>
            ))}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <caption className="sr-only">The 20 acoustic features extracted from the recording</caption>
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                  <th scope="col" className="py-1 pr-2">#</th>
                  <th scope="col" className="py-1 pr-2">Feature</th>
                  <th scope="col" className="py-1 pr-2 text-right">Value</th>
                  <th scope="col" className="py-1 pr-2 text-right">vs. training</th>
                  <th scope="col" className="py-1 text-right">Importance</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((row) => {
                  const top = row.rank != null && row.rank <= 5;
                  return (
                    <tr
                      key={row.name}
                      className={cn("border-t border-border", top && "bg-brand-50 font-medium dark:bg-brand-900/20")}
                    >
                      <td className="py-1 pr-2 text-muted-foreground">{row.rank ?? "–"}</td>
                      <td className="py-1 pr-2">
                        <span className="font-mono">{row.name}</span>
                        {top ? <span className="sr-only"> (top 5 by importance)</span> : null}
                        <span className="block text-xs text-muted-foreground">{FEATURE_DESCRIPTIONS[row.name]}</span>
                      </td>
                      <td className="py-1 pr-2 text-right font-mono tabular-nums">{formatValue(row.value)}</td>
                      <td className="py-1 pr-2 text-right font-mono tabular-nums">
                        {row.z != null ? `${row.z >= 0 ? "+" : ""}${row.z.toFixed(2)} σ` : "–"}
                      </td>
                      <td className="py-1 text-right font-mono tabular-nums">
                        {row.importance != null ? row.importance.toFixed(3) : "–"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {meta ? (
            <p className="text-xs text-muted-foreground">
              {meta.importance ? `Importance: ${meta.importance.label}, ${meta.importance.method}. ` : ""}
              {meta.reference ? `Comparison: ${meta.reference.source}.` : ""}
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function formatValue(value: number): string {
  if (!Number.isFinite(value)) return "–";
  const abs = Math.abs(value);
  return abs >= 100 ? value.toFixed(1) : abs >= 1 ? value.toFixed(3) : value.toFixed(4);
}

function FeatureCharts({ rows, meta }: { rows: FeatureRow[]; meta: ModelFeaturesResponse | null }) {
  const [open, setOpen] = useState(true);
  const hasReference = rows.some((r) => r.z != null);
  const hasImportance = rows.some((r) => r.importance != null);

  const radarRows = useMemo(() => {
    const picked = hasImportance
      ? [...rows].sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99)).slice(0, 10)
      : rows.slice(0, 10);
    return picked.map((r) => ({
      feature: r.name,
      // z-score clipped to ±3 and mapped to 0..100 (50 = training mean)
      deviation: r.z != null ? Math.round(((Math.max(-3, Math.min(3, r.z)) + 3) / 6) * 100) : 0,
    }));
  }, [rows, hasImportance]);

  const importanceRows = useMemo(
    () =>
      [...rows]
        .filter((r) => r.importance != null)
        .sort((a, b) => (b.importance ?? 0) - (a.importance ?? 0))
        .slice(0, 10)
        .map((r) => ({ feature: r.name, importance: Number((r.importance ?? 0).toFixed(4)) })),
    [rows],
  );

  return (
    <div className="card">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls="feature-charts"
        className="flex w-full items-center justify-between gap-2 text-left"
      >
        <span className="text-h3">Feature visualisation</span>
        {open ? <ChevronUp className="h-5 w-5" aria-hidden /> : <ChevronDown className="h-5 w-5" aria-hidden />}
      </button>

      {open ? (
        <div id="feature-charts" className="mt-3 flex flex-col gap-6">
          <figure>
            <figcaption className="mb-1 text-sm font-medium">
              Top {radarRows.length} features vs. training mean
              <span className="block text-xs font-normal text-muted-foreground">
                Radar radius 50 = training mean; 0 / 100 = three standard deviations below / above.
              </span>
            </figcaption>
            {hasReference ? (
              <div className="h-72" role="img" aria-label="Radar chart of standardised feature values">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarRows} outerRadius="70%">
                    <PolarGrid stroke="var(--border)" />
                    <PolarAngleAxis dataKey="feature" tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} />
                    <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar dataKey="deviation" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.3} />
                    <Tooltip formatter={(v: number) => [`${v}`, "deviation index"]} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Training reference statistics are not available.</p>
            )}
          </figure>

          <figure>
            <figcaption className="mb-1 text-sm font-medium">
              Feature importance
              <span className="block text-xs font-normal text-muted-foreground">
                {meta?.importance ? `${meta.importance.label}: ${meta.importance.method}` : "not available"}
              </span>
            </figcaption>
            {hasImportance ? (
              <div className="h-64" role="img" aria-label="Bar chart of feature importance">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={importanceRows} layout="vertical" margin={{ left: 8, right: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis type="number" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="feature" width={72} tick={{ fontSize: 11, fontFamily: "monospace" }} />
                    <Tooltip />
                    <Bar dataKey="importance" fill="#2563eb" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Feature importances are not available.</p>
            )}
          </figure>

          <figure>
            <figcaption className="mb-1 text-sm font-medium">
              Feature heatmap
              <span className="block text-xs font-normal text-muted-foreground">
                Colour = standardised value (blue below, pink above the training mean).
              </span>
            </figcaption>
            <div className="grid grid-cols-4 gap-1 sm:grid-cols-5" role="img" aria-label="Heatmap of standardised feature values">
              {rows.map((r) => (
                <div
                  key={r.name}
                  title={`${r.name}: ${formatValue(r.value)}${r.z != null ? ` (${r.z.toFixed(2)} σ)` : ""}`}
                  className="flex flex-col items-center rounded p-2 text-center"
                  style={{ background: heatColor(r.z) }}
                >
                  <span className="font-mono text-[11px]">{r.name}</span>
                  <span className="text-[11px] tabular-nums">{r.z != null ? `${r.z >= 0 ? "+" : ""}${r.z.toFixed(1)}σ` : formatValue(r.value)}</span>
                </div>
              ))}
            </div>
          </figure>
        </div>
      ) : null}
    </div>
  );
}

/** Diverging colour: blue for below-mean, pink for above-mean, neutral at the mean. */
function heatColor(z: number | null): string {
  if (z == null) return "var(--muted)";
  const t = Math.max(-1, Math.min(1, z / 3));
  const alpha = Math.abs(t) * 0.75 + 0.1;
  return t < 0 ? `rgba(37, 99, 235, ${alpha})` : `rgba(219, 39, 119, ${alpha})`;
}
