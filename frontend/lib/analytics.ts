/** Pure helpers for the analytics page: date ranges, bucketing and formatting. */

import type { AnalyticsFilters } from "@/store/analyticsStore";
import type { GenderFilter, PredictionSummary } from "@/types/api";

export const DAY_MS = 24 * 60 * 60 * 1000;

/** Local calendar date (yyyy-mm-dd) of a timestamp. */
export function toDateKey(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

export function startOfDay(date: Date): Date {
  const copy = new Date(date);
  copy.setHours(0, 0, 0, 0);
  return copy;
}

/** Parses a yyyy-mm-dd string as a local date (null when invalid). */
export function parseDateKey(key: string | null): Date | null {
  if (!key) return null;
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(key);
  if (!match) return null;
  const date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  return Number.isNaN(date.getTime()) ? null : date;
}

export interface DateWindow {
  /** Inclusive start (start of day) or null for "all time". */
  from: Date | null;
  /** Inclusive end (end of day). */
  to: Date;
}

/** Resolves the filter presets into a concrete window. */
export function resolveWindow(filters: AnalyticsFilters, now = new Date()): DateWindow {
  const today = startOfDay(now);
  const endOfToday = new Date(today.getTime() + DAY_MS - 1);
  switch (filters.range) {
    case "7d":
      return { from: new Date(today.getTime() - 6 * DAY_MS), to: endOfToday };
    case "30d":
      return { from: new Date(today.getTime() - 29 * DAY_MS), to: endOfToday };
    case "90d":
      return { from: new Date(today.getTime() - 89 * DAY_MS), to: endOfToday };
    case "custom": {
      const from = parseDateKey(filters.from);
      const toDay = parseDateKey(filters.to);
      const to = toDay ? new Date(toDay.getTime() + DAY_MS - 1) : endOfToday;
      return { from, to };
    }
    default:
      return { from: null, to: endOfToday };
  }
}

export function inWindow(iso: string, window: DateWindow): boolean {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return false;
  if (window.from && t < window.from.getTime()) return false;
  return t <= window.to.getTime();
}

export function matchesGender(item: PredictionSummary, gender: GenderFilter): boolean {
  return gender === "all" || item.gender === gender;
}

export interface TimelinePoint {
  date: string;
  label: string;
  count: number;
}

/**
 * Predictions per day across the window. For "all time" the axis starts at the earliest
 * prediction; empty days are kept so the line shows gaps honestly.
 */
export function buildTimeline(items: PredictionSummary[], window: DateWindow): TimelinePoint[] {
  const counts = new Map<string, number>();
  let earliest: Date | null = null;
  for (const item of items) {
    const date = new Date(item.created_at);
    if (Number.isNaN(date.getTime())) continue;
    const key = toDateKey(date);
    counts.set(key, (counts.get(key) ?? 0) + 1);
    if (!earliest || date < earliest) earliest = date;
  }
  const start = window.from ?? (earliest ? startOfDay(earliest) : startOfDay(window.to));
  const end = startOfDay(window.to);
  const points: TimelinePoint[] = [];
  const formatter = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" });
  for (let t = start.getTime(); t <= end.getTime(); t += DAY_MS) {
    const day = new Date(t);
    const key = toDateKey(day);
    points.push({ date: key, label: formatter.format(day), count: counts.get(key) ?? 0 });
  }
  return points;
}

export interface ConfidenceBucket {
  key: "90-100" | "80-90" | "70-80" | "<70";
  label: string;
  /** Bounds in percent: lower inclusive, upper exclusive (100 inclusive). */
  min: number;
  max: number;
  count: number;
  color: string;
}

export const CONFIDENCE_BUCKETS: Omit<ConfidenceBucket, "count">[] = [
  { key: "90-100", label: "90-100%", min: 90, max: 100, color: "#16a34a" },
  { key: "80-90", label: "80-90%", min: 80, max: 90, color: "#eab308" },
  { key: "70-80", label: "70-80%", min: 70, max: 80, color: "#f97316" },
  { key: "<70", label: "< 70%", min: 0, max: 70, color: "#dc2626" },
];

export function bucketConfidence(items: PredictionSummary[]): ConfidenceBucket[] {
  const buckets = CONFIDENCE_BUCKETS.map((b) => ({ ...b, count: 0 }));
  for (const item of items) {
    const pct = item.confidence * 100;
    const bucket = pct >= 90 ? 0 : pct >= 80 ? 1 : pct >= 70 ? 2 : 3;
    buckets[bucket].count += 1;
  }
  return buckets;
}

/** Count of predictions in the last `days` days vs. the `days` before that. */
export function weekOverWeek(items: PredictionSummary[], days = 7, now = new Date()) {
  const end = startOfDay(now).getTime() + DAY_MS;
  const recentStart = end - days * DAY_MS;
  const previousStart = recentStart - days * DAY_MS;
  let recent = 0;
  let previous = 0;
  for (const item of items) {
    const t = new Date(item.created_at).getTime();
    if (t >= recentStart && t < end) recent += 1;
    else if (t >= previousStart && t < recentStart) previous += 1;
  }
  const changePct = previous > 0 ? Math.round(((recent - previous) / previous) * 100) : null;
  return { recent, previous, changePct };
}

export function formatDate(iso: string, withTime = true): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return new Intl.DateTimeFormat(
    undefined,
    withTime ? { dateStyle: "medium", timeStyle: "short" } : { dateStyle: "medium" },
  ).format(date);
}

export function formatSeconds(value: number | null | undefined, digits = 1): string {
  return value == null ? "–" : `${value.toFixed(digits)} s`;
}

export const MODEL_LABELS: Record<string, string> = {
  MLP: "Neural Network (MLP)",
  RandomForest: "Random Forest",
  SVM: "SVM",
  GradientBoosting: "Gradient Boosting",
  DecisionTree: "Decision Tree",
};

export function modelLabel(key: string): string {
  return MODEL_LABELS[key] ?? key;
}
