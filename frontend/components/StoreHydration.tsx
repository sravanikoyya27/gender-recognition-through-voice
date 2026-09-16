"use client";

import { useEffect } from "react";

import { useAnalyticsStore } from "@/store/analyticsStore";
import { useAuthStore } from "@/store/authStore";

/**
 * Rehydrates persisted Zustand stores after mount.
 *
 * Persisted stores are configured with `skipHydration` so that server-rendered
 * HTML and the first client render agree; this component then loads the saved
 * state from localStorage and flips `hasHydrated`.
 */
export default function StoreHydration() {
  useEffect(() => {
    void useAuthStore.persist.rehydrate();
    void useAnalyticsStore.persist.rehydrate();
  }, []);
  return null;
}
