# frontend/

Next.js 14 (App Router, TypeScript, Tailwind CSS) client for Gender Recognition Through Voice.

## Setup

Requires Node.js 18.17 or newer (Node 22 is used for development).

```bash
cd frontend
npm install
cp .env.example .env.local     # set NEXT_PUBLIC_API_URL to your backend
npm run dev                    # http://localhost:3000
```

Other scripts:

```bash
npm run build        # production build (.next/standalone is used by the Docker image)
npm run start        # serve the production build
npm run lint         # ESLint (next/core-web-vitals + next/typescript)
npm run type-check   # tsc --noEmit
npm run check        # lint + type-check
```

The backend must allow this origin in its `CORS_ORIGINS` setting (default `http://localhost:3000`).

## Versions

| Package | Version | Note |
|---|---|---|
| next / react | 14.2.x / 18.3 | Latest Next 14 line; the pinned libraries below declare React 16-18 peers |
| axios | 1.6.0 | HTTP client |
| zustand | 4.4.4 | State management |
| recharts | 2.10.0 | Charts |
| lucide-react | 0.292.0 | Icons |
| clsx | 2.0.0 | Class-name utility |
| next-auth | 4.24.5 | Optional server-side auth (not wired yet) |
| react-mic | 12.4.6 | Audio recording. 12.7.2 was requested but was never published; 12.4.6 is the last release. It declares a React 16 peer and is installed with `--legacy-peer-deps`. |

## Directory structure

```
frontend/
├── app/                       App Router
│   ├── layout.tsx             Root layout: fonts, Navigation, Footer, global CSS
│   ├── page.tsx               Home page with live API status
│   ├── (auth)/                Route group (no URL segment)
│   │   ├── layout.tsx         Centered card layout
│   │   ├── login/page.tsx     /login
│   │   └── register/page.tsx  /register
│   └── (dashboard)/           Route group with tab navigation
│       ├── layout.tsx
│       ├── dashboard/page.tsx /dashboard  (recorder, upload form, results)
│       └── analytics/page.tsx /analytics  (summary tiles, chart)
├── components/
│   ├── AudioRecorder.tsx      Microphone recorder (contract fixed; behaviour in recording phase)
│   ├── FileUploadForm.tsx     Client-side validated audio upload
│   ├── ResultsDisplay.tsx     Gender badge, confidence, probability bars, metadata
│   ├── ApiStatus.tsx          Calls GET /health and shows backend status
│   ├── StoreHydration.tsx     Rehydrates persisted stores after mount (SSR-safe)
│   ├── Navigation.tsx         Header with active-route highlighting and auth buttons
│   ├── Footer.tsx
│   └── LoadingSpinner.tsx
├── hooks/
│   ├── useAuth.ts             login/register/logout on top of authStore
│   ├── usePrediction.ts       predict(file) on top of predictionStore
│   └── useAnalytics.ts        fetches history and computes a summary
├── services/
│   ├── api.ts                 axios instance, interceptors, endpoint functions
│   └── auth.ts                auth endpoint functions
├── store/
│   ├── authStore.ts           token + user, persisted to localStorage
│   └── predictionStore.ts     status, latest result, session history
├── types/
│   ├── api.ts                 wire types mirroring backend/app/schemas
│   ├── models.ts              domain types, FEATURE_NAMES
│   └── react-mic.d.ts         minimal typings for react-mic
├── lib/utils.ts               cn(), formatPercent(), formatBytes()
├── styles/
│   ├── variables.css          CSS custom properties (light/dark tokens, gender colours)
│   └── globals.css            Tailwind layers, component classes (.card, .btn-primary, .badge-male ...)
├── __tests__/                 Unit tests (runner configured in the testing phase)
├── cypress/e2e/               End-to-end tests (configured in the testing phase)
├── tailwind.config.js         Custom theme: male (blue) / female (pink) / brand palettes, type scale, spacing
├── next.config.js             Standalone output, image optimisation, security headers
├── tsconfig.json              Strict mode, path aliases (@/components, @/types, ...)
└── .env.example               Public vs server-only variables explained
```

Note on routing: the requested `(dashboard)/page.tsx` would resolve to `/` and collide with `app/page.tsx` (Next.js rejects two pages at the same path), so the dashboard lives at `(dashboard)/dashboard/page.tsx` -> `/dashboard`.

## Component conventions

- Server components by default; add `"use client"` only where hooks, browser APIs or event handlers are needed.
- Components receive data through props and never call the API directly; hooks own data fetching and stores own shared state.
- Styling uses Tailwind utilities plus the small set of component classes in `styles/globals.css`. Gender-specific colours come from the `male` / `female` palettes (`text-male`, `bg-female-100`, `.badge-male`).
- Icons come from `lucide-react`; mark decorative icons with `aria-hidden`.

## State management (Zustand)

Stores live in `store/` and are plain hooks created with `create()`.

```ts
// Read a slice (component re-renders only when that slice changes)
const status = usePredictionStore((s) => s.status);

// Call actions
usePredictionStore.getState().reset();
```

- `authStore` uses the `persist` middleware so the JWT survives reloads. It is created with `skipHydration: true`; `components/StoreHydration.tsx` (mounted in the root layout) calls `useAuthStore.persist.rehydrate()` after mount and the store flips `hasHydrated`. Components check `hasHydrated` before rendering auth-dependent UI, which keeps server and first-client render identical.
- `predictionStore` holds the in-flight `status`, the latest `result`, the last `error`, and a session-local `history` (newest first, capped at 50).
- Components should go through the hooks in `hooks/` (`useAuth`, `usePrediction`, `useAnalytics`), which combine store actions with API calls and error normalisation. Errors are always `ApiError` objects (`{ status, message }`) produced by `services/api.ts`.

## Talking to the backend

`services/api.ts` builds one axios instance from `NEXT_PUBLIC_API_URL`, attaches the bearer token from `authStore`, and converts every failure into an `ApiError`. A 401 response clears the session automatically. Endpoint helpers: `getHealth()`, `predictAudio(file)`, `getPredictions()`, plus `login()` / `register()` in `services/auth.ts`.
