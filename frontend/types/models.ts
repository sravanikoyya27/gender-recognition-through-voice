/** Domain models used across the UI. Mirrors backend/app/schemas. */

export type Gender = "male" | "female";

/** Mirrors backend UserResponse (Phase 3B). */
export interface User {
  id: number;
  username: string;
  email: string;
  created_at: string;
}

/** Mirrors backend PredictionResponse (Phase 3C): spec fields plus the aliases the UI reads. */
export interface Prediction {
  /** Database id; null when the prediction was not persisted (anonymous / DB unavailable). */
  prediction_id: number | null;
  gender: Gender;
  predicted_gender: Gender;
  confidence: number;
  confidence_percentage: number;
  probabilities: Record<Gender, number> | null;
  inference_time_ms: number;
  feature_extraction_time_ms: number | null;
  total_time_ms: number | null;
  audio_duration: number;
  duration_seconds: number | null;
  sample_rate: number | null;
  audio_filename: string | null;
  audio_file_path: string | null;
  audio_features: Record<string, number>;
  features: Record<string, number> | null;
  model_used: string;
  model_version: string | null;
  user_id: number | null;
  created_at: string | null;
}

export type PredictionStatus = "idle" | "uploading" | "analyzing" | "success" | "error";

/** The 20 acoustic features consumed by the neural network, in model order. */
export const FEATURE_NAMES = [
  "meanfreq",
  "sd",
  "median",
  "Q25",
  "Q75",
  "IQR",
  "skew",
  "kurt",
  "sp.ent",
  "sfm",
  "mode",
  "centroid",
  "meanfun",
  "minfun",
  "maxfun",
  "meandom",
  "mindom",
  "maxdom",
  "dfrange",
  "modindx",
] as const;

export type FeatureName = (typeof FEATURE_NAMES)[number];
