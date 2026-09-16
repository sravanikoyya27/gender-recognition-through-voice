/** Wire-level types for the FastAPI backend (see backend/app/schemas). */

import type { Gender, Prediction, User } from "./models";

export interface HealthResponse {
  status: "ok" | "degraded";
  app: string;
  version: string;
  environment: string;
  database: "connected" | "unavailable";
  model_loaded: boolean;
  timestamp: string;
}

export type PredictionResponse = Prediction;

// --- Analytics & history (Phase 3D) -----------------------------------------

/** One row of GET /api/v1/me/predictions. */
export interface PredictionSummary {
  id: number;
  gender: Gender;
  confidence: number;
  audio_duration: number | null;
  created_at: string;
  model_used: string;
}

export type HistorySortKey = "date" | "confidence" | "duration";
/** Backend sort_by values: key, or "-key" for descending. */
export type HistorySort = HistorySortKey | `-${HistorySortKey}`;
export type GenderFilter = "all" | Gender;

export interface HistoryQuery {
  limit?: number;
  offset?: number;
  sort_by?: HistorySort;
  gender_filter?: GenderFilter;
}

export interface PredictionHistoryResponse {
  total: number;
  page_size: number;
  page_offset: number;
  sort_by: string;
  gender_filter: string;
  predictions: PredictionSummary[];
}

/** GET /api/v1/me/predictions/{id}: the summary plus stored probabilities and 22 acoustic measurements. */
export interface PredictionDetail extends PredictionSummary {
  inference_time_ms: number | null;
  model_version: string | null;
  probabilities: Record<string, number> | null;
  audio_file_path: string | null;
  audio_features: Record<string, number | null>;
}

export interface GenderDistribution {
  male: number;
  female: number;
  male_percentage: number;
  female_percentage: number;
}

export interface DurationRange {
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
}

export interface DateCount {
  date: string;
  count: number;
}

/** GET /api/v1/me/statistics (all-time aggregates; the server caches them for one hour). */
export interface StatisticsResponse {
  total_predictions: number;
  gender_distribution: GenderDistribution;
  average_confidence: number | null;
  confidence_by_gender: Record<string, number | null>;
  average_audio_duration: number | null;
  duration_range: DurationRange;
  /** Daily counts for the last 30 days, newest first. */
  predictions_by_date: DateCount[];
  generated_at: string;
  cached: boolean;
}

export interface ModelPerformanceEntry {
  name: string;
  model_key: string;
  accuracy: number;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
  inference_time_ms: number | null;
  version: string | null;
  training_date: string | null;
  /** Evaluation (test) set size. */
  total_predictions: number | null;
  evaluation: string;
}

/** GET /api/v1/models/performance (public benchmark metrics). */
export interface ModelPerformanceResponse {
  models: ModelPerformanceEntry[];
  current_model: string;
  note: string;
}

export interface MessageResponse {
  message: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
}

export type UserResponse = User;

export interface RegisterRequest {
  email: string;
  password: string;
  full_name?: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface FeatureImportance {
  model_key: string;
  label: string;
  method: string;
  /** Importance per feature, sorted descending. */
  values: Record<string, number>;
}

export interface FeatureReference {
  source: string;
  mean: Record<string, number>;
  std: Record<string, number>;
}

export interface ModelFeaturesResponse {
  feature_order: string[];
  importance: FeatureImportance | null;
  reference: FeatureReference | null;
  note: string;
}

/** Normalised error surfaced by services/api.ts for any failed request. */
export interface ApiError {
  status: number | null;
  message: string;
  detail?: unknown;
}
