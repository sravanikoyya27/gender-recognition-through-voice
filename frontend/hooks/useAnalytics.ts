"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import {
  deleteMyPrediction,
  getModelPerformance,
  getMyPredictions,
  getMyStatistics,
  toApiError,
} from "@/services/api";
import { isFresh, PAGE_SIZE, useAnalyticsStore } from "@/store/analyticsStore";
import { useAuthStore } from "@/store/authStore";
import type { ApiError, PredictionSummary } from "@/types/api";

/** Page size used to pull the whole history for the charts (backend maximum). */
const HISTORY_PAGE = 100;
/** Upper bound on rows loaded for the charts (10 requests). */
export const HISTORY_CAP = 1000;

/**
 * Loads the analytics page data from the Phase 3D endpoints and keeps it in the analytics store:
 * statistics (all-time, server-cached 1 h), model benchmark metrics, the complete history
 * for the charts, and the current table page (server-side pagination, sort and gender filter).
 * Fetched data is reused for one hour unless a prediction is created or deleted.
 */
export function useAnalytics() {
  const token = useAuthStore((s) => s.token);
  const authHydrated = useAuthStore((s) => s.hasHydrated);
  const store = useAnalyticsStore();
  const { filters, sort, offset, hasHydrated, setResource } = store;
  const debouncedGender = useDebouncedValue(filters.gender, 500);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [deleteError, setDeleteError] = useState<ApiError | null>(null);
  const controllers = useRef<Record<string, AbortController | undefined>>({});

  const ready = authHydrated && hasHydrated && Boolean(token);

  /** Aborts the previous request with the same name and returns a fresh signal. */
  const signalFor = useCallback((name: string) => {
    controllers.current[name]?.abort();
    const controller = new AbortController();
    controllers.current[name] = controller;
    return controller.signal;
  }, []);

  const loadStatistics = useCallback(async () => {
    const signal = signalFor("statistics");
    setResource("statistics", { loading: true, error: null });
    try {
      const data = await getMyStatistics(signal);
      if (!signal.aborted) setResource("statistics", { data, fetchedAt: Date.now(), loading: false });
    } catch (err) {
      if (!signal.aborted) setResource("statistics", { loading: false, error: toApiError(err) });
    }
  }, [setResource, signalFor]);

  const loadModels = useCallback(async () => {
    const signal = signalFor("models");
    setResource("models", { loading: true, error: null });
    try {
      const data = await getModelPerformance(signal);
      if (!signal.aborted) setResource("models", { data, fetchedAt: Date.now(), loading: false });
    } catch (err) {
      if (!signal.aborted) setResource("models", { loading: false, error: toApiError(err) });
    }
  }, [setResource, signalFor]);

  const loadHistory = useCallback(async () => {
    const signal = signalFor("history");
    setResource("history", { loading: true, error: null });
    try {
      const items: PredictionSummary[] = [];
      let total = 0;
      for (let page = 0; page * HISTORY_PAGE < HISTORY_CAP; page += 1) {
        const response = await getMyPredictions(
          { limit: HISTORY_PAGE, offset: page * HISTORY_PAGE, sort_by: "-date" },
          signal,
        );
        total = response.total;
        items.push(...response.predictions);
        if (items.length >= total || response.predictions.length === 0) break;
      }
      if (!signal.aborted) {
        setResource("history", {
          data: { items, total, truncated: items.length < total },
          fetchedAt: Date.now(),
          loading: false,
        });
      }
    } catch (err) {
      if (!signal.aborted) setResource("history", { loading: false, error: toApiError(err) });
    }
  }, [setResource, signalFor]);

  const loadPage = useCallback(async () => {
    const signal = signalFor("page");
    setResource("page", { loading: true, error: null });
    try {
      const data = await getMyPredictions(
        { limit: PAGE_SIZE, offset, sort_by: sort, gender_filter: debouncedGender },
        signal,
      );
      if (!signal.aborted) setResource("page", { data, fetchedAt: Date.now(), loading: false });
    } catch (err) {
      if (!signal.aborted) setResource("page", { loading: false, error: toApiError(err) });
    }
  }, [setResource, signalFor, offset, sort, debouncedGender]);

  // Model benchmarks are public: load them even before sign-in.
  // (No `loading` guard: an aborted request never clears the flag, and signalFor() already
  // cancels the previous in-flight request, which also keeps StrictMode's double effects safe.)
  useEffect(() => {
    if (!authHydrated) return;
    if (!isFresh(useAnalyticsStore.getState().models.fetchedAt)) void loadModels();
  }, [authHydrated, loadModels]);

  // Per-user resources: fetch once per hour (or after invalidation).
  useEffect(() => {
    if (!ready) return;
    const state = useAnalyticsStore.getState();
    if (!isFresh(state.statistics.fetchedAt)) void loadStatistics();
    if (!isFresh(state.history.fetchedAt)) void loadHistory();
  }, [ready, loadStatistics, loadHistory]);

  // The table page depends on pagination, sort and the (debounced) gender filter.
  useEffect(() => {
    if (!ready) return;
    const page = useAnalyticsStore.getState().page;
    const sameQuery =
      page.data &&
      page.data.page_offset === offset &&
      page.data.sort_by === sort &&
      page.data.gender_filter === debouncedGender;
    if (sameQuery && isFresh(page.fetchedAt)) return;
    void loadPage();
  }, [ready, loadPage, offset, sort, debouncedGender]);

  // Abort in-flight requests when the page unmounts.
  useEffect(() => {
    const active = controllers.current;
    return () => {
      Object.values(active).forEach((c) => c?.abort());
    };
  }, []);

  const deletePrediction = useCallback(
    async (id: number): Promise<boolean> => {
      setDeletingId(id);
      setDeleteError(null);
      try {
        await deleteMyPrediction(id);
        useAnalyticsStore.getState().removePrediction(id);
        // The server dropped its statistics cache; refresh the aggregates and refill the page.
        await Promise.all([loadStatistics(), loadPage()]);
        return true;
      } catch (err) {
        setDeleteError(toApiError(err));
        return false;
      } finally {
        setDeletingId(null);
      }
    },
    [loadStatistics, loadPage],
  );

  const refreshAll = useCallback(() => {
    void loadStatistics();
    void loadModels();
    void loadHistory();
    void loadPage();
  }, [loadStatistics, loadModels, loadHistory, loadPage]);

  return {
    ready,
    isAuthenticated: Boolean(token),
    hasHydrated: authHydrated && hasHydrated,
    filters,
    setFilters: store.setFilters,
    resetFilters: store.resetFilters,
    sort,
    toggleSort: store.toggleSort,
    offset,
    setOffset: store.setOffset,
    statistics: store.statistics,
    models: store.models,
    history: store.history,
    page: store.page,
    reload: { statistics: loadStatistics, models: loadModels, history: loadHistory, page: loadPage, all: refreshAll },
    deletePrediction,
    deletingId,
    deleteError,
  };
}
