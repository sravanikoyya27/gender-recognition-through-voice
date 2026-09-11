import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import type { User } from "@/types/models";

interface AuthState {
  token: string | null;
  user: User | null;
  /** True once persisted state has been read from localStorage (avoids SSR hydration mismatches). */
  hasHydrated: boolean;
  isAuthenticated: () => boolean;
  setSession: (token: string, user: User | null) => void;
  setUser: (user: User | null) => void;
  logout: () => void;
  setHasHydrated: (value: boolean) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      hasHydrated: false,
      isAuthenticated: () => Boolean(get().token),
      setSession: (token, user) => set({ token, user }),
      setUser: (user) => set({ user }),
      logout: () => set({ token: null, user: null }),
      setHasHydrated: (value) => set({ hasHydrated: value }),
    }),
    {
      name: "vg-auth",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ token: state.token, user: state.user }),
      // Hydrate from localStorage only after mount (see components/StoreHydration.tsx) so the
      // first client render matches the server-rendered HTML.
      skipHydration: true,
      onRehydrateStorage: () => (state) => state?.setHasHydrated(true),
    },
  ),
);
