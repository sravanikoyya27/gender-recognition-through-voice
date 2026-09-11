/** Domain models used across the UI. Mirrors backend/app/schemas. */

export type Gender = "male" | "female";

export interface User {
  id: number;
  email: string;
  full_name: string | null;
  is_active: boolean;
  created_at: string;
}

export interface Prediction {
  id: number | null;
  predicted_gender: Gender;
  confidence: number;
  probabilities: Record<Gender, number> | null;
  features: Record<string, number> | null;
  model_version: string | null;
  duration_seconds: number | null;
  sample_rate: number | null;
  audio_filename: string | null;
  created_at: string | null;
}

export type PredictionStatus = "idle" | "uploading" | "analyzing" | "success" | "error";

export interface AnalyticsSummary {
  total: number;
  byGender: Record<Gender, number>;
  averageConfidence: number;
}

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
