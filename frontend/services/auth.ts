import { API_V1, api } from "@/services/api";
import type { LoginRequest, RegisterRequest, TokenResponse, UserResponse } from "@/types/api";

export async function login(payload: LoginRequest): Promise<TokenResponse> {
  const { data } = await api.post<TokenResponse>(`${API_V1}/auth/login`, payload);
  return data;
}

export async function register(payload: RegisterRequest): Promise<UserResponse> {
  const { data } = await api.post<UserResponse>(`${API_V1}/auth/register`, payload);
  return data;
}
