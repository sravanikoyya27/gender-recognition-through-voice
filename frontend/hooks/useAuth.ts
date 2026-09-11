"use client";

import { useCallback, useState } from "react";

import * as authService from "@/services/auth";
import { toApiError } from "@/services/api";
import { useAuthStore } from "@/store/authStore";
import type { ApiError, LoginRequest, RegisterRequest } from "@/types/api";

export function useAuth() {
  const { token, user, hasHydrated, setSession, logout } = useAuthStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const login = useCallback(
    async (payload: LoginRequest): Promise<boolean> => {
      setLoading(true);
      setError(null);
      try {
        const { access_token } = await authService.login(payload);
        setSession(access_token, null);
        return true;
      } catch (err) {
        setError(toApiError(err));
        return false;
      } finally {
        setLoading(false);
      }
    },
    [setSession],
  );

  const register = useCallback(async (payload: RegisterRequest): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      await authService.register(payload);
      return true;
    } catch (err) {
      setError(toApiError(err));
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    token,
    user,
    isAuthenticated: Boolean(token),
    hasHydrated,
    loading,
    error,
    login,
    register,
    logout,
  };
}
