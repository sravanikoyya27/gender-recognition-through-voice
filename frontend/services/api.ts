import axios, { AxiosError, type AxiosInstance } from "axios";

import { useAuthStore } from "@/store/authStore";
import type {
  ApiError,
  HealthResponse,
  PredictionListResponse,
  PredictionResponse,
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

/** Convert any thrown value into a predictable ApiError. */
export function toApiError(error: unknown): ApiError {
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

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>("/health");
  return data;
}

export async function predictAudio(file: File | Blob, filename?: string): Promise<PredictionResponse> {
  const form = new FormData();
  const name = filename ?? (file instanceof File ? file.name : "recording.webm");
  form.append("file", file, name);
  const { data } = await api.post<PredictionResponse>(`${API_V1}/predictions`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function getPredictions(): Promise<PredictionListResponse> {
  const { data } = await api.get<PredictionListResponse>(`${API_V1}/predictions`);
  return data;
}
