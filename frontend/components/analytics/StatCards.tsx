"use client";

import { Clock, Percent, TrendingDown, TrendingUp, Users } from "lucide-react";
import { memo, type ReactNode } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";

import Skeleton from "@/components/Skeleton";
import { formatSeconds, weekOverWeek } from "@/lib/analytics";
import { cn, formatPercent } from "@/lib/utils";
import type { ApiError, PredictionSummary, StatisticsResponse } from "@/types/api";

interface StatCardsProps {
  statistics: StatisticsResponse | null;
  loading: boolean;
  error: ApiError | null;
  /** Complete history (for the week-over-week trend). */
  history: PredictionSummary[] | null;
  onRetry: () => void;
}

const ACCENTS = {
  blue: { border: "border-l-male-500", text: "text-male-700 dark:text-male-300" },
  pink: { border: "border-l-female-500", text: "text-female-700 dark:text-female-300" },
  green: { border: "border-l-green-500", text: "text-green-700 dark:text-green-400" },
  orange: { border: "border-l-orange-500", text: "text-orange-700 dark:text-orange-400" },
};

function StatCard({
  title,
  accent,
  icon,
  children,
}: {
  title: string;
  accent: keyof typeof ACCENTS;
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className={cn("card flex flex-col gap-2 border-l-4", ACCENTS[accent].border)}>
      <h3 className="flex items-center justify-between text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {title}
        <span className={ACCENTS[accent].text} aria-hidden>
          {icon}
        </span>
      </h3>
      {children}
    </div>
  );
}

function StatCards({ statistics, loading, error, history, onRetry }: StatCardsProps) {
  if (error && !statistics) {
    return (
      <div role="alert" className="card flex flex-wrap items-center justify-between gap-3 border-red-200 text-sm">
        <span>
          Statistics could not be loaded{error.status ? ` (HTTP ${error.status})` : ""}: {error.message}
        </span>
        <button type="button" className="btn-secondary" onClick={onRetry}>
          Retry
        </button>
      </div>
    );
  }

  if (!statistics || (loading && !statistics)) {
    return (
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-busy="true" aria-label="Loading statistics">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="card flex flex-col gap-3">
            <Skeleton className="h-3 w-24" />
            <Skeleton className="h-9 w-20" />
            <Skeleton className="h-3 w-32" />
          </div>
        ))}
      </div>
    );
  }

  const { total_predictions: total, gender_distribution: gd, duration_range: dr } = statistics;
  const trend = history ? weekOverWeek(history) : null;
  const avgConfidence = statistics.average_confidence;
  const maleConf = statistics.confidence_by_gender.male;
  const femaleConf = statistics.confidence_by_gender.female;
  const pieData = [
    { name: "Male", value: gd.male, fill: "var(--color-male)" },
    { name: "Female", value: gd.female, fill: "var(--color-female)" },
  ];

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <StatCard title="Total predictions" accent="blue" icon={<Users className="h-4 w-4" />}>
        <p className="font-mono text-4xl font-semibold tabular-nums text-male-700 dark:text-male-300">{total}</p>
        <p className="flex items-center gap-1 text-sm text-muted-foreground">
          {trend ? (
            trend.changePct != null ? (
              <>
                {trend.changePct >= 0 ? (
                  <TrendingUp className="h-4 w-4 text-green-600" aria-hidden />
                ) : (
                  <TrendingDown className="h-4 w-4 text-red-600" aria-hidden />
                )}
                <span>
                  {trend.changePct >= 0 ? "+" : ""}
                  {trend.changePct}% vs. previous 7 days ({trend.recent} vs. {trend.previous})
                </span>
              </>
            ) : (
              <span>{trend.recent} in the last 7 days</span>
            )
          ) : (
            <span>All time</span>
          )}
        </p>
      </StatCard>

      <StatCard title="Male / Female split" accent="pink" icon={<Users className="h-4 w-4" />}>
        <div className="flex items-center gap-3">
          <div className="h-16 w-16 shrink-0" role="img" aria-label={`Male ${gd.male_percentage}%, female ${gd.female_percentage}%`}>
            {total > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={pieData} dataKey="value" innerRadius="60%" outerRadius="100%" stroke="none" isAnimationActive={false}>
                    {pieData.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full w-full rounded-full border-4 border-muted" />
            )}
          </div>
          <dl className="text-sm">
            <div className="flex items-center gap-2">
              <dt className="h-2.5 w-2.5 rounded-full bg-male" aria-hidden />
              <dd>
                <span className="font-medium text-male-700 dark:text-male-300">Male</span> {gd.male_percentage.toFixed(0)}% ({gd.male})
              </dd>
            </div>
            <div className="flex items-center gap-2">
              <dt className="h-2.5 w-2.5 rounded-full bg-female" aria-hidden />
              <dd>
                <span className="font-medium text-female-700 dark:text-female-300">Female</span> {gd.female_percentage.toFixed(0)}% ({gd.female})
              </dd>
            </div>
          </dl>
        </div>
      </StatCard>

      <StatCard title="Average confidence" accent="green" icon={<Percent className="h-4 w-4" />}>
        <p className="font-mono text-4xl font-semibold tabular-nums text-green-700 dark:text-green-400">
          {avgConfidence != null ? formatPercent(avgConfidence, 1) : "–"}
        </p>
        <p className="text-sm text-muted-foreground">
          {maleConf != null || femaleConf != null ? (
            <>
              Male {maleConf != null ? formatPercent(maleConf, 1) : "–"} · Female{" "}
              {femaleConf != null ? formatPercent(femaleConf, 1) : "–"}
            </>
          ) : (
            "No predictions yet"
          )}
        </p>
      </StatCard>

      <StatCard title="Average duration" accent="orange" icon={<Clock className="h-4 w-4" />}>
        <p className="font-mono text-4xl font-semibold tabular-nums text-orange-700 dark:text-orange-400">
          {statistics.average_audio_duration != null ? `${statistics.average_audio_duration.toFixed(1)} s` : "–"}
        </p>
        <p className="text-sm text-muted-foreground">
          {dr.min != null && dr.max != null ? `Range ${formatSeconds(dr.min)} – ${formatSeconds(dr.max)}` : "No recordings yet"}
        </p>
      </StatCard>
    </div>
  );
}

export default memo(StatCards);
