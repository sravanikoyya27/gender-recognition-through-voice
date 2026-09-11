/** Wire-level types for the FastAPI backend (see backend/app/schemas). */

import type { Prediction, User } from "./models";

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

export interface PredictionListResponse {
  items: Prediction[];
  total: number;
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

/** Normalised error surfaced by services/api.ts for any failed request. */
export interface ApiError {
  status: number | null;
  message: string;
  detail?: unknown;
}
