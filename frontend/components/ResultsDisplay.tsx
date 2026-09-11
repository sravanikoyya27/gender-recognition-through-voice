"use client";

import { AlertCircle, Sparkles } from "lucide-react";

import LoadingSpinner from "@/components/LoadingSpinner";
import { cn, formatPercent } from "@/lib/utils";
import type { ApiError } from "@/types/api";
import type { Prediction, PredictionStatus } from "@/types/models";

export interface ResultsDisplayProps {
  status: PredictionStatus;
  result: Prediction | null;
  error: ApiError | null;
  className?: string;
}

export default function ResultsDisplay({ status, result, error, className }: ResultsDisplayProps) {
  return (
    <section aria-live="polite" className={cn("card flex flex-col gap-4", className)}>
      <h2 className="text-h3">Result</h2>

      {status === "idle" && !result ? (
        <p className="text-sm text-muted-foreground">
          Record or upload a voice sample to see the prediction here.
        </p>
      ) : null}

      {status === "uploading" || status === "analyzing" ? (
        <LoadingSpinner
          size="lg"
          label={status === "uploading" ? "Uploading audio..." : "Extracting features and running the model..."}
        />
      ) : null}

      {status === "error" && error ? (
        <div role="alert" className="flex items-start gap-2 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <div>
            <p className="font-medium">Prediction failed{error.status ? ` (HTTP ${error.status})` : ""}</p>
            <p>{error.message}</p>
          </div>
        </div>
      ) : null}

      {status === "success" && result ? <ResultCard result={result} /> : null}
    </section>
  );
}

function ResultCard({ result }: { result: Prediction }) {
  const isMale = result.predicted_gender === "male";
  const probabilities = result.probabilities ?? {
    male: isMale ? result.confidence : 1 - result.confidence,
    female: isMale ? 1 - result.confidence : result.confidence,
  };

  return (
    <div className="animate-fade-in flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <Sparkles className={cn("h-6 w-6", isMale ? "text-male" : "text-female")} aria-hidden />
        <span className={isMale ? "badge-male" : "badge-female"}>
          {isMale ? "Male" : "Female"}
        </span>
        <span className="text-sm text-muted-foreground">
          {formatPercent(result.confidence)} confidence
        </span>
      </div>

      <dl className="grid gap-2 text-sm">
        <ProbabilityBar label="Male" value={probabilities.male} colorClass="bg-male" />
        <ProbabilityBar label="Female" value={probabilities.female} colorClass="bg-female" />
      </dl>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground sm:grid-cols-4">
        {result.audio_filename ? <Meta label="File" value={result.audio_filename} /> : null}
        {result.duration_seconds != null ? (
          <Meta label="Duration" value={`${result.duration_seconds.toFixed(2)} s`} />
        ) : null}
        {result.sample_rate ? <Meta label="Sample rate" value={`${result.sample_rate} Hz`} /> : null}
        {result.model_version ? <Meta label="Model" value={result.model_version} /> : null}
      </dl>
    </div>
  );
}

function ProbabilityBar({ label, value, colorClass }: { label: string; value: number; colorClass: string }) {
  return (
    <div>
      <div className="flex justify-between">
        <dt>{label}</dt>
        <dd className="font-mono">{formatPercent(value)}</dd>
      </div>
      <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full transition-all", colorClass)} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="uppercase tracking-wide">{label}</dt>
      <dd className="truncate text-foreground" title={value}>
        {value}
      </dd>
    </div>
  );
}
