"use client";

import { CalendarRange, FilterX, SlidersHorizontal } from "lucide-react";
import { useEffect, useId, useState } from "react";

import { cn } from "@/lib/utils";
import { type AnalyticsFilters as Filters, type RangePreset } from "@/store/analyticsStore";
import type { GenderFilter } from "@/types/api";

const RANGE_PRESETS: { value: RangePreset; label: string }[] = [
  { value: "7d", label: "Last 7 days" },
  { value: "30d", label: "Last 30 days" },
  { value: "90d", label: "Last 90 days" },
  { value: "all", label: "All time" },
  { value: "custom", label: "Custom" },
];

const GENDER_OPTIONS: { value: GenderFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "male", label: "Male only" },
  { value: "female", label: "Female only" },
];

interface AnalyticsFiltersProps {
  filters: Filters;
  onChange: (patch: Partial<Filters>) => void;
  onReset: () => void;
  disabled?: boolean;
}

export default function AnalyticsFilters({ filters, onChange, onReset, disabled }: AnalyticsFiltersProps) {
  const id = useId();
  // The slider updates a local value immediately and commits to the store after 500 ms of rest.
  const [threshold, setThreshold] = useState(filters.threshold);
  useEffect(() => setThreshold(filters.threshold), [filters.threshold]);
  useEffect(() => {
    if (threshold === filters.threshold) return;
    const timer = window.setTimeout(() => onChange({ threshold }), 500);
    return () => window.clearTimeout(timer);
  }, [threshold, filters.threshold, onChange]);

  const isDefault =
    filters.range === "30d" && filters.gender === "all" && filters.threshold === 0 && !filters.from && !filters.to;

  return (
    <section aria-labelledby={`${id}-title`} className="card flex flex-col gap-4">
      <div className="flex items-center justify-between gap-2">
        <h2 id={`${id}-title`} className="flex items-center gap-2 text-h3">
          <SlidersHorizontal className="h-5 w-5 text-brand" aria-hidden />
          Filters
        </h2>
        <button type="button" className="btn-secondary" onClick={onReset} disabled={disabled || isDefault}>
          <FilterX className="h-4 w-4" aria-hidden />
          Reset
        </button>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {/* Date range */}
        <fieldset className="flex min-w-0 flex-col gap-2">
          <legend className="label flex items-center gap-1">
            <CalendarRange className="h-4 w-4" aria-hidden />
            Date range
          </legend>
          <div className="flex flex-wrap gap-1" role="group" aria-label="Date range presets">
            {RANGE_PRESETS.map((preset) => (
              <button
                key={preset.value}
                type="button"
                disabled={disabled}
                aria-pressed={filters.range === preset.value}
                onClick={() => onChange({ range: preset.value })}
                className={cn(
                  "rounded border px-3 py-1 text-sm transition-colors",
                  filters.range === preset.value
                    ? "border-brand bg-brand text-white"
                    : "border-border bg-card hover:bg-muted",
                )}
              >
                {preset.label}
              </button>
            ))}
          </div>
          {filters.range === "custom" ? (
            <div className="flex flex-wrap items-end gap-2">
              <div>
                <label htmlFor={`${id}-from`} className="text-xs text-muted-foreground">
                  From
                </label>
                <input
                  id={`${id}-from`}
                  type="date"
                  className="input"
                  disabled={disabled}
                  value={filters.from ?? ""}
                  max={filters.to ?? undefined}
                  onChange={(e) => onChange({ from: e.target.value || null })}
                />
              </div>
              <div>
                <label htmlFor={`${id}-to`} className="text-xs text-muted-foreground">
                  To
                </label>
                <input
                  id={`${id}-to`}
                  type="date"
                  className="input"
                  disabled={disabled}
                  value={filters.to ?? ""}
                  min={filters.from ?? undefined}
                  onChange={(e) => onChange({ to: e.target.value || null })}
                />
              </div>
            </div>
          ) : null}
        </fieldset>

        {/* Gender */}
        <fieldset className="flex flex-col gap-2">
          <legend className="label">Gender</legend>
          <div className="flex flex-wrap gap-1" role="group" aria-label="Gender filter">
            {GENDER_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                disabled={disabled}
                aria-pressed={filters.gender === option.value}
                onClick={() => onChange({ gender: option.value })}
                className={cn(
                  "rounded border px-3 py-1 text-sm transition-colors",
                  filters.gender === option.value
                    ? option.value === "male"
                      ? "border-male bg-male text-white"
                      : option.value === "female"
                        ? "border-female bg-female text-white"
                        : "border-brand bg-brand text-white"
                    : "border-border bg-card hover:bg-muted",
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
        </fieldset>

        {/* Confidence threshold */}
        <div className="flex flex-col gap-2">
          <label htmlFor={`${id}-threshold`} className="label flex items-center justify-between">
            <span>Confidence threshold</span>
            <span className="font-mono text-sm tabular-nums">{threshold}%</span>
          </label>
          <input
            id={`${id}-threshold`}
            type="range"
            min={0}
            max={100}
            step={5}
            value={threshold}
            disabled={disabled}
            onChange={(e) => setThreshold(Number(e.target.value))}
            aria-valuetext={`${threshold} percent`}
            className="w-full accent-brand"
          />
          <p className="text-xs text-muted-foreground">
            Predictions below the threshold are highlighted in the table and the confidence chart.
          </p>
        </div>
      </div>
    </section>
  );
}
