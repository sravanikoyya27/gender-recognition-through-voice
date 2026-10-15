import axios, { AxiosError, type AxiosInstance, type AxiosProgressEvent } from "axios";

import { useAuthStore } from "@/store/authStore";
import type {
  ApiError,
  HealthResponse,
  HistoryQuery,
  MessageResponse,
  ModelFeaturesResponse,
  ModelPerformanceResponse,
  PredictionDetail,
  PredictionHistoryResponse,
  PredictionResponse,
  StatisticsResponse,
} from "@/types/api";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const API_V1 = "/api/v1";

/** Shared axios instance: base URL, JSON defaults, bearer token, error normalisation. */
export const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30_000,
  headers: { Accept: "application/json" },
});

api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ detail?: unknown }>) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
    }
    return Promise.reject(toApiError(error));
  },
);

export const CANCELLED = "cancelled";

/** Convert any thrown value into a predictable ApiError. */
export function toApiError(error: unknown): ApiError {
  if (axios.isCancel(error)) {
    return { status: null, message: "Upload cancelled", detail: CANCELLED };
  }
  if (axios.isAxiosError<{ detail?: unknown }>(error)) {
    const detail = error.response?.data?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : error.response
          ? `Request failed with status ${error.response.status}`
          : error.code === "ECONNABORTED"
            ? "The request timed out"
            : "Could not reach the API server";
    return { status: error.response?.status ?? null, message, detail };
  }
  if (isApiError(error)) return error;
  return { status: null, message: error instanceof Error ? error.message : "Unknown error" };
}

export function isApiError(value: unknown): value is ApiError {
  return typeof value === "object" && value !== null && "message" in value && "status" in value;
}

// --- Endpoints ---------------------------------------------------------------

/** Readiness check: 200 when healthy, 503 with the same body when degraded (both resolve here). */
export async function getHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>("/health", {
    validateStatus: (code) => code === 200 || code === 503,
  });
  return data;
}

export interface UploadOptions {
  /** Progress callback (loaded/total bytes) for a progress bar. */
  onUploadProgress?: (event: AxiosProgressEvent) => void;
  /** Abort signal to cancel the request. */
  signal?: AbortSignal;
}

export async function predictAudio(
  file: File | Blob,
  filename?: string,
  options: UploadOptions = {},
): Promise<PredictionResponse> {
  const form = new FormData();
  const name = filename ?? (file instanceof File ? file.name : "recording.wav");
  form.append("audio", file, name); // backend field name (Phase 3C); "file" is accepted as an alias
  const { data } = await api.post<PredictionResponse>(`${API_V1}/predictions`, form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: options.onUploadProgress,
    signal: options.signal,
    timeout: 120_000,
  });
  return data;
}

/** Feature order, measured importances and training reference statistics (public, static). */
export async function getModelFeatures(): Promise<ModelFeaturesResponse> {
  const { data } = await api.get<ModelFeaturesResponse>(`${API_V1}/models/features`);
  return data;
}

// --- Analytics & history (Phase 3D endpoints, authenticated) -----------------

/** Paginated history of the signed-in user (limit 1-100, sort "-date" by default). */
export async function getMyPredictions(
  query: HistoryQuery = {},
  signal?: AbortSignal,
): Promise<PredictionHistoryResponse> {
  const { data } = await api.get<PredictionHistoryResponse>(`${API_V1}/me/predictions`, {
    params: query,
    signal,
  });
  return data;
}

/** One prediction with probabilities and all stored acoustic measurements. */
export async function getMyPrediction(id: number, signal?: AbortSignal): Promise<PredictionDetail> {
  const { data } = await api.get<PredictionDetail>(`${API_V1}/me/predictions/${id}`, { signal });
  return data;
}

export async function deleteMyPrediction(id: number): Promise<MessageResponse> {
  const { data } = await api.delete<MessageResponse>(`${API_V1}/me/predictions/${id}`);
  return data;
}

/** All-time statistics of the signed-in user (server-side cache: one hour). */
export async function getMyStatistics(signal?: AbortSignal): Promise<StatisticsResponse> {
  const { data } = await api.get<StatisticsResponse>(`${API_V1}/me/statistics`, { signal });
  return data;
}

/** Benchmark metrics of the trained models (public). */
export async function getModelPerformance(signal?: AbortSignal): Promise<ModelPerformanceResponse> {
  const { data } = await api.get<ModelPerformanceResponse>(`${API_V1}/models/performance`, { signal });
  return data;
}
