"use client";

import { RefreshCw } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import LazyVisible from "@/components/LazyVisible";
import LoadingSpinner from "@/components/LoadingSpinner";
import AnalyticsFilters from "@/components/analytics/AnalyticsFilters";
import { ChartCard, ConfidenceHistogram, GenderDonut, ModelPerformanceChart, TimelineChart } from "@/components/analytics/charts";
import PredictionDetail from "@/components/analytics/PredictionDetail";
import PredictionsTable from "@/components/analytics/PredictionsTable";
import StatCards from "@/components/analytics/StatCards";
import { HISTORY_CAP, useAnalytics } from "@/hooks/useAnalytics";
import { bucketConfidence, buildTimeline, inWindow, matchesGender, resolveWindow } from "@/lib/analytics";

const RANGE_LABEL = { "7d": "last 7 days", "30d": "last 30 days", "90d": "last 90 days", all: "all time", custom: "custom range" };

export default function AnalyticsPage() {
  return (
    <Suspense fallback={<LoadingSpinner label="Loading analytics..." />}>
      <AnalyticsContent />
    </Suspense>
  );
}

function AnalyticsContent() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const analytics = useAnalytics();
  const { filters, setFilters, hasHydrated, isAuthenticated, statistics, models, history, page, reload } = analytics;
  // The Suspense boundary can hydrate after the persisted stores have already rehydrated, so the
  // first client render must not depend on store state: show the spinner until mounted.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  // ?gender=male|female (from "Similar predictions" on the dashboard) preselects the gender filter once.
  const appliedGenderParam = useRef<string | null>(null);
  useEffect(() => {
    const gender = searchParams.get("gender");
    if (!hasHydrated || !gender || appliedGenderParam.current === gender) return;
    appliedGenderParam.current = gender;
    if (gender === "male" || gender === "female") setFilters({ gender });
  }, [searchParams, hasHydrated, setFilters]);

  // ?prediction=<id> (from "Share link" or the table's View action) opens the detail modal.
  const predictionParam = searchParams.get("prediction");
  const detailId = predictionParam && /^\d+$/.test(predictionParam) ? Number(predictionParam) : null;
  const closeDetail = useCallback(() => {
    const params = new URLSearchParams(searchParams.toString());
    params.delete("prediction");
    const query = params.toString();
    router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
  }, [router, pathname, searchParams]);

  // Charts: the complete history filtered client-side by date window and gender.
  const dateWindow = useMemo(() => resolveWindow(filters), [filters]);
  const filteredHistory = useMemo(
    () => (history.data?.items ?? []).filter((p) => inWindow(p.created_at, dateWindow) && matchesGender(p, filters.gender)),
    [history.data, dateWindow, filters.gender],
  );
  const timeline = useMemo(() => buildTimeline(filteredHistory, dateWindow), [filteredHistory, dateWindow]);
  const genderCounts = useMemo(
    () => ({
      male: filteredHistory.filter((p) => p.gender === "male").length,
      female: filteredHistory.filter((p) => p.gender === "female").length,
    }),
    [filteredHistory],
  );
  const buckets = useMemo(() => bucketConfidence(filteredHistory), [filteredHistory]);

  const rangeText = `${RANGE_LABEL[filters.range]}${filters.gender !== "all" ? `, ${filters.gender} only` : ""}`;
  const chartDescription = history.data
    ? `${filteredHistory.length} of your ${history.data.total} predictions (${rangeText}).${
        history.data.truncated ? ` Only the ${HISTORY_CAP} most recent predictions are loaded for the charts.` : ""
      }`
    : undefined;

  if (!mounted || !hasHydrated) {
    return <LoadingSpinner label="Loading analytics..." />;
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-h1">Your Analytics</h1>
          <p className="text-sm text-muted-foreground">
            History and statistics of your stored predictions, straight from your account.
          </p>
        </div>
        {isAuthenticated ? (
          <button type="button" className="btn-secondary" onClick={reload.all} aria-label="Refresh analytics">
            <RefreshCw className={statistics.loading || history.loading ? "h-4 w-4 animate-spin" : "h-4 w-4"} aria-hidden />
            Refresh
          </button>
        ) : null}
      </header>

      {!isAuthenticated ? (
        <div className="card text-sm">
          <p>
            Predictions are stored per account.{" "}
            <Link href="/login" className="text-brand underline-offset-4 hover:underline">
              Sign in
            </Link>{" "}
            to see your history and statistics.
          </p>
        </div>
      ) : (
        <>
          <AnalyticsFilters filters={filters} onChange={setFilters} onReset={analytics.resetFilters} />

          <StatCards
            statistics={statistics.data}
            loading={statistics.loading}
            error={statistics.error}
            history={history.data?.items ?? null}
            onRetry={reload.statistics}
          />
          {statistics.data ? (
            <p className="-mt-4 text-xs text-muted-foreground">
              Cards show all-time totals (server cache: 1 hour
              {statistics.data.cached ? ", served from cache" : ""}). Charts follow the filters.
            </p>
          ) : null}

          <ChartCard
            title="Predictions timeline"
            description={chartDescription ?? "Predictions per day."}
            loading={history.loading && !history.data}
            error={history.error && !history.data ? history.error : null}
            onRetry={reload.history}
          >
            <LazyVisible>
              <TimelineChart data={timeline} />
            </LazyVisible>
          </ChartCard>

          <div className="grid gap-6 lg:grid-cols-2">
            <ChartCard
              title="Gender distribution"
              description={chartDescription}
              loading={history.loading && !history.data}
              error={history.error && !history.data ? history.error : null}
              onRetry={reload.history}
            >
              <LazyVisible>
                <GenderDonut male={genderCounts.male} female={genderCounts.female} />
              </LazyVisible>
            </ChartCard>

            <ChartCard
              title="Confidence distribution"
              description={
                <>
                  {chartDescription}
                  {filters.threshold > 0 ? ` Buckets entirely below the ${filters.threshold}% threshold are dimmed.` : ""}
                </>
              }
              loading={history.loading && !history.data}
              error={history.error && !history.data ? history.error : null}
              onRetry={reload.history}
            >
              <LazyVisible>
                <ConfidenceHistogram buckets={buckets} threshold={filters.threshold} />
              </LazyVisible>
            </ChartCard>
          </div>
        </>
      )}

      <ChartCard
        title="Model performance"
        description={
          models.data
            ? `Test accuracy on the historical voice.csv benchmark split (${models.data.models[0]?.total_predictions ?? "?"} rows). ${models.data.note} Current production model: ${models.data.current_model}.`
            : "Benchmark accuracy of the trained models."
        }
        loading={models.loading && !models.data}
        error={models.error && !models.data ? models.error : null}
        onRetry={reload.models}
      >
        {models.data ? (
          <LazyVisible>
            <ModelPerformanceChart data={models.data} />
          </LazyVisible>
        ) : null}
      </ChartCard>

      {isAuthenticated ? (
        <PredictionsTable
          page={page.data}
          loading={page.loading}
          error={page.error}
          sort={analytics.sort}
          onSort={analytics.toggleSort}
          offset={analytics.offset}
          onOffset={analytics.setOffset}
          threshold={filters.threshold}
          onDelete={analytics.deletePrediction}
          deletingId={analytics.deletingId}
          deleteError={analytics.deleteError}
          onRetry={reload.page}
          filtered={filters.gender !== "all"}
        />
      ) : null}

      {detailId != null && isAuthenticated ? <PredictionDetail id={detailId} onClose={closeDetail} /> : null}
    </div>
  );
}
