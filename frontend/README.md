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
│       └── analytics/page.tsx /analytics  (filters, statistic cards, charts, history table, detail modal)
├── components/
│   ├── AudioRecorder.tsx      Microphone recorder: MediaRecorder + Web Audio waveform, timer, playback, 1-10 s limits, WAV output
│   ├── FileUploadForm.tsx     Drag-and-drop upload: validation (extension, MIME, magic bytes), progress, speed, ETA, cancel, retry
│   ├── ResultsDisplay.tsx     Hero gender, animated confidence gauge/bar, metadata, sortable feature table, radar/importance/heatmap charts, actions
│   ├── analytics/             Analytics page parts: AnalyticsFilters, StatCards, charts (timeline, donut, histogram, model accuracy), PredictionsTable, PredictionDetail
│   ├── ApiStatus.tsx          Calls GET /health and shows backend status
│   ├── ConfirmDialog.tsx      Accessible modal confirmation (focus trap, Escape)
│   ├── LazyVisible.tsx        Renders children once scrolled near the viewport (chart lazy loading)
│   ├── Skeleton.tsx           Loading placeholder block
│   ├── StoreHydration.tsx     Rehydrates persisted stores after mount (SSR-safe)
│   ├── Navigation.tsx         Header with active-route highlighting and auth buttons
│   ├── Footer.tsx
│   └── LoadingSpinner.tsx
├── hooks/
│   ├── useAuth.ts             login/register/logout on top of authStore
│   ├── usePrediction.ts       predict(file) on top of predictionStore
│   ├── useAnalytics.ts        loads statistics, model metrics, history and the table page into analyticsStore
│   ├── useDebouncedValue.ts   debounce helper (filters → fetch)
│   ├── useModelFeatures.ts    feature importances / reference stats (cached per session)
│   └── useCountUp.ts          animated number
├── services/
│   ├── api.ts                 axios instance, interceptors, endpoint functions
│   └── auth.ts                auth endpoint functions
├── store/
│   ├── authStore.ts           token + user, persisted to localStorage
│   ├── predictionStore.ts     status, latest result, session history
│   ├── recorderStore.ts       recorder state
│   ├── uploadStore.ts         upload state
│   └── analyticsStore.ts      analytics filters (persisted), sort/page, cached responses (1 h)
├── types/
│   ├── api.ts                 wire types mirroring backend/app/schemas
│   ├── models.ts              domain types, FEATURE_NAMES
│   └── react-mic.d.ts         minimal typings for react-mic
├── lib/utils.ts               cn(), formatPercent(), formatBytes()
├── lib/analytics.ts           date windows, timeline/confidence bucketing, formatting for the analytics page
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

## Audio recorder

`components/AudioRecorder.tsx` records with the native `MediaRecorder` API (react-mic is installed but not used: it is unmaintained, targets React 16 and bundles analytics). State lives in `store/recorderStore.ts` (`isRecording`, `recordingTime`, `audioBlob`, `isPlaying`, `error`, `warning`, `permission`), the mechanics in `hooks/useAudioRecorder.ts`:

- Start (red, pulsing while recording), Stop (only while recording), Play/Pause and Clear after a take; timer in `mm:ss.t` updated every 100 ms; live green waveform from an `AnalyserNode` on a canvas, static peaks afterwards.
- Limits: shorter than `minDuration` (1 s) is rejected with an error, `maxDuration` (10 s) stops the recording automatically, with warnings as the limits approach.
- Errors: permission denied, no microphone, unsupported browser, recording failure.
- Output: the take is decoded with Web Audio and re-encoded as **16 kHz mono 16-bit WAV** (`lib/audio.ts`) before `onRecordingComplete(blob)` is called, because the backend decodes with libsndfile and cannot read WebM/Opus.
- Accessibility: ARIA labels on all controls, a polite live region for timer and status, Space toggles recording while the recorder has focus.

```tsx
<AudioRecorder onRecordingComplete={handleRecording} maxDuration={10} minDuration={1} />
```

## File upload

`components/FileUploadForm.tsx` accepts `.wav` and `.mp3` files up to 10 MB by drag-and-drop (dashed zone, highlighted while dragging, keyboard-operable) or through the file input. State lives in `store/uploadStore.ts` (`selectedFile`, `status`, `uploadProgress`, `uploadSpeed`, `estimatedTime`, `error`, `retryable`).

- Validation (`lib/upload.ts`): extension, reported MIME type, and the file header read with `FileReader` (`RIFF…WAVE` for WAV, `ID3` tag or MPEG frame sync for MP3), each with a specific message.
- Upload: `services/api.ts` `predictAudio()` posts `multipart/form-data` to `POST /api/v1/predictions` with Axios `onUploadProgress` and an `AbortSignal`; the form shows a green progress bar, percentage, bytes, smoothed speed and time remaining, a Cancel button while uploading, and Retry after a failed upload. The result goes through `usePrediction` into `predictionStore`, so `ResultsDisplay` shows it; "Upload another file" clears the form and keeps the result visible.

## Results display

`components/ResultsDisplay.tsx` renders the shared prediction state (`status`, `result`, `error` from `predictionStore`): a loading spinner while uploading/analysing, an error card, and on success a two-column layout (single column on mobile):

- Hero: "MALE"/"FEMALE" in blue/pink with an icon (colour is never the only cue), scale-in animation, and a screen-reader summary.
- Confidence: count-up percentage, SVG gauge and progress bar coloured green (> 90 %), yellow (70–90 %) or red (< 70 %).
- Metadata: duration, recording time (client receipt time, the API response carries none), model ("Neural Network (MLP)"), inference time, sample rate, file name.
- Features: collapsible sortable table of the 20 values (top 5 by importance highlighted, standardised deviation from the training mean), radar chart of the top 10 (z-scores clipped to ±3σ), importance bar chart and a heatmap. Importances and reference statistics come from `GET /api/v1/models/features`, which serves the measured Phase 2C Random Forest importances and Phase 2B training statistics; the charts state that provenance and degrade when the artifacts are absent (the MLP itself has no native importances).
- Actions: Try another recording (resets recorder, upload form and result), Download audio (when the analysed audio is still in the browser), Save (predictions are stored server-side when signed in; the button shows the saved id), Share (copies a link to the prediction in your history), Similar predictions (history filtered by gender).

## Analytics page

`app/(dashboard)/analytics/page.tsx` shows the signed-in user's stored predictions using the Phase 3D endpoints only (no client-side mock data):

- **Data** (`hooks/useAnalytics.ts` → `store/analyticsStore.ts`): `GET /api/v1/me/statistics` for the four statistic cards (all-time aggregates, cached by the server for one hour), `GET /api/v1/models/performance` for the model chart (public), `GET /api/v1/me/predictions?limit=10&offset=…&sort_by=…&gender_filter=…` for the table (server-side pagination, sort and gender filter), and the same endpoint paged with `limit=100` (up to 1000 rows) to load the complete history for the charts. Responses are reused for one hour; creating or deleting a prediction invalidates them.
- **Filters** (persisted to localStorage): date range presets (7/30/90 days, all time, custom from/to) and the gender filter drive the charts client-side; the gender filter is also sent to the server for the table (debounced 500 ms). The confidence-threshold slider highlights rows and dims histogram buckets below it; it never hides data.
- **Cards**: total predictions (with a real week-over-week trend computed from the history), male/female split with a mini donut, average confidence (with the per-gender averages from the API), average duration with the min–max range.
- **Charts** (recharts, memoised, rendered lazily when scrolled into view): predictions per day, gender donut with counts and percentages, confidence histogram (90-100 / 80-90 / 70-80 / <70), and test accuracy per model with the current production model highlighted. The model chart states that the accuracies were measured on the historical voice.csv test split and that live-audio accuracy is unvalidated.
- **Table**: Date / Gender / Confidence / Duration / Model, sortable by date, confidence and duration (the backend's sort keys), 10 rows per page, View (opens the detail modal via `?prediction=<id>`, the same link "Share" copies on the dashboard) and Delete (confirmation dialog, then `DELETE /api/v1/me/predictions/{id}`; the cards and table refresh). `?gender=male|female` (the dashboard's "Similar predictions" link) preselects the gender filter.

## Talking to the backend

`services/api.ts` builds one axios instance from `NEXT_PUBLIC_API_URL`, attaches the bearer token from `authStore`, and converts every failure into an `ApiError`. A 401 response clears the session automatically. Endpoint helpers: `getHealth()`, `predictAudio(file)`, `getModelFeatures()`, `getMyPredictions(query)`, `getMyPrediction(id)`, `deleteMyPrediction(id)`, `getMyStatistics()`, `getModelPerformance()`, plus `login()` / `register()` in `services/auth.ts`.
