"use client";

import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import LoadingSpinner from "@/components/LoadingSpinner";
import { formatDate, formatSeconds, modelLabel } from "@/lib/analytics";
import { cn, formatPercent } from "@/lib/utils";
import { getMyPrediction, toApiError } from "@/services/api";
import type { ApiError, PredictionDetail as Detail } from "@/types/api";
import { FEATURE_NAMES } from "@/types/models";

interface PredictionDetailProps {
  id: number;
  onClose: () => void;
}

function formatFeature(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "–";
  const abs = Math.abs(value);
  return abs >= 100 ? value.toFixed(1) : abs >= 1 ? value.toFixed(3) : value.toFixed(4);
}

/** Modal with one stored prediction (GET /api/v1/me/predictions/{id}); opened by `?prediction=<id>`. */
export default function PredictionDetail({ id, onClose }: PredictionDetailProps) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    setDetail(null);
    setError(null);
    getMyPrediction(id, controller.signal)
      .then(setDetail)
      .catch((err: unknown) => {
        if (!controller.signal.aborted) setError(toApiError(err));
      });
    return () => controller.abort();
  }, [id]);

  useEffect(() => {
    closeRef.current?.focus();
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const isMale = detail?.gender === "male";
  const extra = detail
    ? Object.keys(detail.audio_features).filter((k) => !(FEATURE_NAMES as readonly string[]).includes(k))
    : [];

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/50 p-4 sm:items-center"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div role="dialog" aria-modal="true" aria-labelledby="prediction-detail-title" className="card w-full max-w-2xl animate-scale-in shadow-lg">
        <div className="flex items-start justify-between gap-4">
          <h2 id="prediction-detail-title" className="text-h3">
            Prediction #{id}
          </h2>
          <button ref={closeRef} type="button" onClick={onClose} className="btn-secondary px-2 py-1" aria-label="Close details">
            <X className="h-4 w-4" aria-hidden />
          </button>
        </div>

        {error ? (
          <p role="alert" className="mt-4 text-sm text-red-600">
            {error.status === 404
              ? "This prediction does not exist (it may have been deleted)."
              : error.status === 403
                ? "This prediction belongs to another account."
                : `Could not load the prediction${error.status ? ` (HTTP ${error.status})` : ""}: ${error.message}`}
          </p>
        ) : !detail ? (
          <div className="mt-6 flex justify-center">
            <LoadingSpinner label="Loading prediction..." />
          </div>
        ) : (
          <div className="mt-4 flex flex-col gap-4">
            <div className="flex flex-wrap items-center gap-3">
              <span className={cn(isMale ? "badge-male" : "badge-female", "text-base")}>{isMale ? "Male" : "Female"}</span>
              <span className="font-mono text-2xl font-semibold tabular-nums">{formatPercent(detail.confidence, 2)}</span>
              <span className="text-sm text-muted-foreground">confidence</span>
            </div>

            <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
              {(
                [
                  ["Recorded", formatDate(detail.created_at)],
                  ["Duration", formatSeconds(detail.audio_duration, 2)],
                  ["Model", modelLabel(detail.model_used)],
                  ["Model version", detail.model_version ?? "–"],
                  ["Inference time", detail.inference_time_ms != null ? `${detail.inference_time_ms.toFixed(1)} ms` : "–"],
                  ["Audio file", detail.audio_file_path ?? "–"],
                  [
                    "P(male) / P(female)",
                    detail.probabilities
                      ? `${formatPercent(detail.probabilities.male ?? 0, 2)} / ${formatPercent(detail.probabilities.female ?? 0, 2)}`
                      : "–",
                  ],
                ] as [string, string][]
              ).map(([k, v]) => (
                <div key={k} className="flex justify-between gap-4 border-b border-border py-1">
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className="truncate text-right font-medium" title={v}>
                    {v}
                  </dd>
                </div>
              ))}
            </dl>

            <div>
              <h3 className="text-sm font-medium">Acoustic features</h3>
              <p className="text-xs text-muted-foreground">
                The 20 model inputs, plus {extra.length ? extra.join(" and ") : "no"} auxiliary measurements stored with the prediction.
              </p>
              <ul className="mt-2 grid grid-cols-2 gap-1 text-xs sm:grid-cols-4">
                {[...FEATURE_NAMES, ...extra].map((name) => (
                  <li key={name} className="flex justify-between gap-2 rounded bg-muted px-2 py-1">
                    <span className="font-mono">{name}</span>
                    <span className="tabular-nums">{formatFeature(detail.audio_features[name] ?? null)}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
