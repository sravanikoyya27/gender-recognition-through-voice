import { API_V1, api } from "@/services/api";
import type {
  LoginRequest,
  MessageResponse,
  RegisterRequest,
  RegisterResponse,
  TokenResponse,
  UserResponse,
} from "@/types/api";

/** Phase 3B endpoints under /api/v1/auth. Errors surface as ApiError via the shared client. */

export async function login(payload: LoginRequest): Promise<TokenResponse> {
  const { data } = await api.post<TokenResponse>(`${API_V1}/auth/login`, payload);
  return data;
}

export async function register(payload: RegisterRequest): Promise<RegisterResponse> {
  const { data } = await api.post<RegisterResponse>(`${API_V1}/auth/register`, payload);
  return data;
}

/** The account that owns the current bearer token (401 when signed out). */
export async function getCurrentUser(): Promise<UserResponse> {
  const { data } = await api.get<UserResponse>(`${API_V1}/auth/me`);
  return data;
}

/** Tokens are stateless; the endpoint only logs the event. The client must discard the token. */
export async function logout(): Promise<MessageResponse> {
  const { data } = await api.post<MessageResponse>(`${API_V1}/auth/logout`);
  return data;
}
