"use client";

import { useCallback, useEffect, useState } from "react";

import { getPredictions, toApiError } from "@/services/api";
import type { ApiError } from "@/types/api";
import type { AnalyticsSummary, Prediction } from "@/types/models";

export function summarise(items: Prediction[]): AnalyticsSummary {
  const byGender = { male: 0, female: 0 };
  let confidenceSum = 0;
  for (const item of items) {
    byGender[item.predicted_gender] += 1;
    confidenceSum += item.confidence;
  }
  return {
    total: items.length,
    byGender,
    averageConfidence: items.length ? confidenceSum / items.length : 0,
  };
}

export function useAnalytics() {
  const [items, setItems] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getPredictions();
      setItems(data.items);
    } catch (err) {
      setError(toApiError(err));
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { items, summary: summarise(items), loading, error, refresh };
}
