"use client";

import { useCallback, useState } from "react";

import * as authService from "@/services/auth";
import { toApiError } from "@/services/api";
import { useAnalyticsStore } from "@/store/analyticsStore";
import { useAuthStore } from "@/store/authStore";
import type { ApiError, LoginRequest, RegisterRequest } from "@/types/api";
import type { User } from "@/types/models";

export function useAuth() {
  const { token, user, hasHydrated, setSession, setUser, logout: clearSession } = useAuthStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  /** Exchange credentials for a JWT, store it, then load the account behind it. */
  const login = useCallback(
    async (payload: LoginRequest): Promise<boolean> => {
      setLoading(true);
      setError(null);
      try {
        const { access_token } = await authService.login(payload);
        setSession(access_token, null);
        try {
          setUser(await authService.getCurrentUser()); // authenticated request with the new token
        } catch {
          // The token is valid even if the profile lookup fails; the UI just lacks the username.
        }
        return true;
      } catch (err) {
        setError(toApiError(err));
        return false;
      } finally {
        setLoading(false);
      }
    },
    [setSession, setUser],
  );

  /** Create the account. Returns the created user (no session is opened; sign in next). */
  const register = useCallback(async (payload: RegisterRequest): Promise<User | null> => {
    setLoading(true);
    setError(null);
    try {
      const { user: created } = await authService.register(payload);
      return created;
    } catch (err) {
      setError(toApiError(err));
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  /** Tell the backend (best effort; JWTs are stateless) and drop the local session and cached data. */
  const logout = useCallback(async (): Promise<void> => {
    const hadToken = Boolean(useAuthStore.getState().token);
    if (hadToken) {
      try {
        await authService.logout();
      } catch {
        // Expired or already-invalid token: the local sign-out still proceeds.
      }
    }
    clearSession();
    useAnalyticsStore.getState().invalidate(); // per-user analytics must not survive the session
  }, [clearSession]);

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
