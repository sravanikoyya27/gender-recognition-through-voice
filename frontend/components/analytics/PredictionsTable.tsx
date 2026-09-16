"use client";

import { ArrowDown, ArrowUp, ArrowUpDown, ChevronLeft, ChevronRight, Eye, Trash2 } from "lucide-react";
import Link from "next/link";
import { memo, useState } from "react";

import ConfirmDialog from "@/components/ConfirmDialog";
import Skeleton from "@/components/Skeleton";
import { formatDate, formatSeconds, modelLabel } from "@/lib/analytics";
import { cn, formatPercent } from "@/lib/utils";
import { PAGE_SIZE, sortKeyOf } from "@/store/analyticsStore";
import type { ApiError, HistorySort, HistorySortKey, PredictionHistoryResponse } from "@/types/api";

interface PredictionsTableProps {
  page: PredictionHistoryResponse | null;
  loading: boolean;
  error: ApiError | null;
  sort: HistorySort;
  onSort: (key: HistorySortKey) => void;
  offset: number;
  onOffset: (offset: number) => void;
  /** Confidence threshold in percent; rows below it are highlighted. */
  threshold: number;
  onDelete: (id: number) => Promise<boolean>;
  deletingId: number | null;
  deleteError: ApiError | null;
  onRetry: () => void;
  /** True when a gender filter is active (changes the empty-state copy). */
  filtered: boolean;
}

const SORTABLE: { key: HistorySortKey; label: string }[] = [
  { key: "date", label: "Date" },
  { key: "confidence", label: "Confidence" },
  { key: "duration", label: "Duration" },
];

function SortHeader({
  column,
  sort,
  onSort,
  className,
}: {
  column: { key: HistorySortKey; label: string };
  sort: HistorySort;
  onSort: (key: HistorySortKey) => void;
  className?: string;
}) {
  const current = sortKeyOf(sort);
  const active = current.key === column.key;
  return (
    <th
      scope="col"
      aria-sort={active ? (current.desc ? "descending" : "ascending") : "none"}
      className={cn("px-3 py-2 text-left text-xs font-medium uppercase tracking-wide text-muted-foreground", className)}
    >
      <button
        type="button"
        onClick={() => onSort(column.key)}
        className={cn("inline-flex items-center gap-1 rounded hover:text-foreground", active && "text-foreground")}
        aria-label={`Sort by ${column.label.toLowerCase()}${active ? (current.desc ? ", currently descending" : ", currently ascending") : ""}`}
      >
        {column.label}
        {active ? (
          current.desc ? (
            <ArrowDown className="h-3.5 w-3.5" aria-hidden />
          ) : (
            <ArrowUp className="h-3.5 w-3.5" aria-hidden />
          )
        ) : (
          <ArrowUpDown className="h-3.5 w-3.5 opacity-50" aria-hidden />
        )}
      </button>
    </th>
  );
}

function PredictionsTable({
  page,
  loading,
  error,
  sort,
  onSort,
  offset,
  onOffset,
  threshold,
  onDelete,
  deletingId,
  deleteError,
  onRetry,
  filtered,
}: PredictionsTableProps) {
  const [pendingDelete, setPendingDelete] = useState<number | null>(null);

  const total = page?.total ?? 0;
  const rows = page?.predictions ?? [];
  const pageIndex = Math.floor(offset / PAGE_SIZE);
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const first = total === 0 ? 0 : offset + 1;
  const last = Math.min(offset + rows.length, total);

  async function confirmDelete() {
    if (pendingDelete == null) return;
    const ok = await onDelete(pendingDelete);
    if (ok) setPendingDelete(null);
  }

  return (
    <section className="card" aria-labelledby="predictions-table-title">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="predictions-table-title" className="text-h3">
          Predictions
        </h2>
        <p className="text-sm text-muted-foreground" aria-live="polite">
          {page ? (total > 0 ? `Showing ${first}–${last} of ${total}` : "No rows") : ""}
        </p>
      </div>

      {error && !page ? (
        <div role="alert" className="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm">
          <span>
            History could not be loaded{error.status ? ` (HTTP ${error.status})` : ""}: {error.message}
          </span>
          <button type="button" className="btn-secondary" onClick={onRetry}>
            Retry
          </button>
        </div>
      ) : !page && loading ? (
        <div className="mt-4 flex flex-col gap-2" aria-busy="true" aria-label="Loading predictions">
          {Array.from({ length: 5 }, (_, i) => (
            <Skeleton key={i} className="h-10 w-full" />
          ))}
        </div>
      ) : total === 0 ? (
        <div className="mt-4 rounded border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
          {filtered ? (
            <p>No predictions match the current gender filter.</p>
          ) : (
            <p>
              No predictions yet.{" "}
              <Link href="/dashboard" className="text-brand underline-offset-4 hover:underline">
                Record or upload a voice sample
              </Link>{" "}
              to get started.
            </p>
          )}
        </div>
      ) : (
        <>
          <div className={cn("mt-4 overflow-x-auto transition-opacity", loading && "opacity-60")} aria-busy={loading}>
            <table className="w-full min-w-[640px] text-sm">
              <caption className="sr-only">
                Your predictions, sorted by {sortKeyOf(sort).key} {sortKeyOf(sort).desc ? "descending" : "ascending"}
                {threshold > 0 ? `; rows below ${threshold}% confidence are highlighted` : ""}
              </caption>
              <thead className="border-b border-border">
                <tr>
                  <SortHeader column={SORTABLE[0]} sort={sort} onSort={onSort} />
                  <th scope="col" className="px-3 py-2 text-left text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Gender
                  </th>
                  <SortHeader column={SORTABLE[1]} sort={sort} onSort={onSort} className="text-right" />
                  <SortHeader column={SORTABLE[2]} sort={sort} onSort={onSort} className="text-right" />
                  <th scope="col" className="px-3 py-2 text-left text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Model
                  </th>
                  <th scope="col" className="px-3 py-2 text-right text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const below = threshold > 0 && row.confidence * 100 < threshold;
                  return (
                    <tr
                      key={row.id}
                      className={cn("border-b border-border last:border-0", below && "bg-orange-100/70 dark:bg-orange-950/40")}
                      aria-label={below ? `Prediction ${row.id}, below the ${threshold}% confidence threshold` : undefined}
                    >
                      <td className="whitespace-nowrap px-3 py-2">
                        <time dateTime={row.created_at}>{formatDate(row.created_at)}</time>
                      </td>
                      <td className="px-3 py-2">
                        <span className={cn(row.gender === "male" ? "badge-male" : "badge-female", "py-0.5 text-xs")}>
                          {row.gender === "male" ? "Male" : "Female"}
                        </span>
                      </td>
                      <td className={cn("px-3 py-2 text-right font-mono tabular-nums", below && "text-orange-700 dark:text-orange-400")}>
                        {formatPercent(row.confidence, 1)}
                        {below ? <span className="sr-only"> (below threshold)</span> : null}
                      </td>
                      <td className="px-3 py-2 text-right font-mono tabular-nums">{formatSeconds(row.audio_duration, 2)}</td>
                      <td className="whitespace-nowrap px-3 py-2">{modelLabel(row.model_used)}</td>
                      <td className="px-3 py-2">
                        <div className="flex justify-end gap-1">
                          <Link
                            href={`/analytics?prediction=${row.id}`}
                            scroll={false}
                            className="btn-secondary px-2 py-1"
                            aria-label={`View details of prediction ${row.id}`}
                          >
                            <Eye className="h-4 w-4" aria-hidden />
                            <span className="hidden sm:inline">View</span>
                          </Link>
                          <button
                            type="button"
                            onClick={() => setPendingDelete(row.id)}
                            disabled={deletingId != null}
                            className="btn-secondary px-2 py-1 text-red-600 hover:bg-red-50 dark:hover:bg-red-950/30"
                            aria-label={`Delete prediction ${row.id}`}
                          >
                            <Trash2 className="h-4 w-4" aria-hidden />
                            <span className="hidden sm:inline">Delete</span>
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <nav className="mt-4 flex flex-wrap items-center justify-between gap-2" aria-label="Pagination">
            <p className="text-sm text-muted-foreground">
              Page {pageIndex + 1} of {pageCount}
            </p>
            <div className="flex gap-1">
              <button
                type="button"
                className="btn-secondary px-2"
                onClick={() => onOffset(Math.max(0, offset - PAGE_SIZE))}
                disabled={offset === 0 || loading}
                aria-label="Previous page"
              >
                <ChevronLeft className="h-4 w-4" aria-hidden />
                Previous
              </button>
              <button
                type="button"
                className="btn-secondary px-2"
                onClick={() => onOffset(offset + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= total || loading}
                aria-label="Next page"
              >
                Next
                <ChevronRight className="h-4 w-4" aria-hidden />
              </button>
            </div>
          </nav>
        </>
      )}

      <ConfirmDialog
        open={pendingDelete != null}
        title="Delete this prediction?"
        confirmLabel="Delete"
        destructive
        busy={deletingId != null}
        error={deleteError?.message ?? null}
        onConfirm={() => void confirmDelete()}
        onCancel={() => setPendingDelete(null)}
      >
        Prediction #{pendingDelete} and its stored acoustic features will be removed from your history. This cannot be
        undone.
      </ConfirmDialog>
    </section>
  );
}

export default memo(PredictionsTable);
