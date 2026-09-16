"use client";

import { useEffect, useState } from "react";

import { getModelFeatures } from "@/services/api";
import type { ModelFeaturesResponse } from "@/types/api";

let cached: ModelFeaturesResponse | null = null;
let inflight: Promise<ModelFeaturesResponse> | null = null;

/** Fetches the (static) model feature metadata once per session and shares it across components. */
export function useModelFeatures() {
  const [data, setData] = useState<ModelFeaturesResponse | null>(cached);
  const [loading, setLoading] = useState(!cached);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (cached) return;
    let cancelled = false;
    inflight ??= getModelFeatures();
    inflight
      .then((response) => {
        cached = response;
        if (!cancelled) setData(response);
      })
      .catch((err: unknown) => {
        inflight = null;
        if (!cancelled) setError(err instanceof Error ? err.message : "Feature metadata unavailable");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { data, loading, error };
}
