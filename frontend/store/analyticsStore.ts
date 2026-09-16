import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import type {
  ApiError,
  GenderFilter,
  HistorySort,
  HistorySortKey,
  ModelPerformanceResponse,
  PredictionHistoryResponse,
  PredictionSummary,
  StatisticsResponse,
} from "@/types/api";

/** Analytics responses are reused for one hour (matches the backend statistics cache TTL). */
export const ANALYTICS_CACHE_TTL_MS = 60 * 60 * 1000;
export const PAGE_SIZE = 10;

export type RangePreset = "7d" | "30d" | "90d" | "all" | "custom";

export interface AnalyticsFilters {
  range: RangePreset;
  /** ISO dates (yyyy-mm-dd) used when `range` is "custom". */
  from: string | null;
  to: string | null;
  gender: GenderFilter;
  /** Confidence threshold in percent (0-100); rows below it are highlighted. */
  threshold: number;
}

export const DEFAULT_FILTERS: AnalyticsFilters = {
  range: "30d",
  from: null,
  to: null,
  gender: "all",
  threshold: 0,
};

/** A fetched resource with its age, so the hook can decide whether to refetch. */
export interface Resource<T> {
  data: T | null;
  fetchedAt: number | null;
  loading: boolean;
  error: ApiError | null;
}

const emptyResource = <T>(): Resource<T> => ({ data: null, fetchedAt: null, loading: false, error: null });

export interface HistoryData {
  /** Every prediction of the user (newest first), fetched page by page for the charts. */
  items: PredictionSummary[];
  /** True when more rows exist than the fetch cap allows. */
  truncated: boolean;
  total: number;
}

export interface AnalyticsState {
  hasHydrated: boolean;
  filters: AnalyticsFilters;
  sort: HistorySort;
  offset: number;

  statistics: Resource<StatisticsResponse>;
  models: Resource<ModelPerformanceResponse>;
  history: Resource<HistoryData>;
  page: Resource<PredictionHistoryResponse>;

  setFilters: (patch: Partial<AnalyticsFilters>) => void;
  resetFilters: () => void;
  /** Click on a sortable header: same key toggles direction, a new key sorts descending. */
  toggleSort: (key: HistorySortKey) => void;
  setOffset: (offset: number) => void;

  setResource: <K extends "statistics" | "models" | "history" | "page">(
    key: K,
    patch: Partial<AnalyticsState[K]>,
  ) => void;
  /** Mark cached data stale (after a new prediction or a deletion) so the next visit refetches. */
  invalidate: () => void;
  /** Drop a deleted prediction from the cached lists without waiting for a refetch. */
  removePrediction: (id: number) => void;
  setHasHydrated: (value: boolean) => void;
}

export function isFresh(fetchedAt: number | null, now = Date.now()): boolean {
  return fetchedAt != null && now - fetchedAt < ANALYTICS_CACHE_TTL_MS;
}

export function sortKeyOf(sort: HistorySort): { key: HistorySortKey; desc: boolean } {
  const desc = sort.startsWith("-");
  return { key: (desc ? sort.slice(1) : sort) as HistorySortKey, desc };
}

export const useAnalyticsStore = create<AnalyticsState>()(
  persist(
    (set) => ({
      hasHydrated: false,
      filters: DEFAULT_FILTERS,
      sort: "-date",
      offset: 0,

      statistics: emptyResource(),
      models: emptyResource(),
      history: emptyResource(),
      page: emptyResource(),

      setFilters: (patch) =>
        set((state) => ({
          filters: { ...state.filters, ...patch },
          // A different gender filter changes the result set: back to the first page.
          offset: patch.gender !== undefined && patch.gender !== state.filters.gender ? 0 : state.offset,
        })),
      resetFilters: () => set({ filters: DEFAULT_FILTERS, offset: 0 }),
      toggleSort: (key) =>
        set((state) => {
          const current = sortKeyOf(state.sort);
          const desc = current.key === key ? !current.desc : true;
          return { sort: desc ? `-${key}` : key, offset: 0 };
        }),
      setOffset: (offset) => set({ offset: Math.max(0, offset) }),

      setResource: (key, patch) => set((state) => ({ [key]: { ...state[key], ...patch } })),
      invalidate: () =>
        set((state) => ({
          statistics: { ...state.statistics, fetchedAt: null },
          history: { ...state.history, fetchedAt: null },
          page: { ...state.page, fetchedAt: null },
        })),
      removePrediction: (id) =>
        set((state) => ({
          history: state.history.data
            ? {
                ...state.history,
                data: {
                  ...state.history.data,
                  items: state.history.data.items.filter((p) => p.id !== id),
                  total: Math.max(0, state.history.data.total - 1),
                },
              }
            : state.history,
          page: state.page.data
            ? {
                ...state.page,
                data: {
                  ...state.page.data,
                  predictions: state.page.data.predictions.filter((p) => p.id !== id),
                  total: Math.max(0, state.page.data.total - 1),
                },
              }
            : state.page,
        })),
      setHasHydrated: (value) => set({ hasHydrated: value }),
    }),
    {
      name: "vg-analytics-filters",
      storage: createJSONStorage(() => localStorage),
      // Only the user's filter preferences persist; fetched data is session-only.
      partialize: (state) => ({ filters: state.filters, sort: state.sort }),
      skipHydration: true,
      onRehydrateStorage: () => (state) => state?.setHasHydrated(true),
    },
  ),
);
