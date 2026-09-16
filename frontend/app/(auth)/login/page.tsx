"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { type FormEvent, Suspense, useState } from "react";

import LoadingSpinner from "@/components/LoadingSpinner";
import { useAuth } from "@/hooks/useAuth";
import { safeNextPath } from "@/lib/authValidation";

export default function LoginPage() {
  return (
    <Suspense fallback={<LoadingSpinner label="Loading..." />}>
      <LoginForm />
    </Suspense>
  );
}

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login, loading, error } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // /login?registered=<username> comes from a successful registration; ?next=/path from a protected page.
  const registered = searchParams.get("registered");
  const next = safeNextPath(searchParams.get("next"));

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const ok = await login({ email: email.trim(), password });
    if (ok) router.push(next);
  }

  return (
    <form onSubmit={handleSubmit} className="card flex flex-col gap-4">
      <h1 className="text-h2">Sign in</h1>

      {registered ? (
        <p role="status" className="rounded border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-800 dark:border-green-900 dark:bg-green-950/40 dark:text-green-300">
          Account <strong>{registered}</strong> created. Sign in with your email and password.
        </p>
      ) : null}

      <div>
        <label htmlFor="email" className="label">
          Email
        </label>
        <input
          id="email"
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="input"
        />
      </div>

      <div>
        <label htmlFor="password" className="label">
          Password
        </label>
        <input
          id="password"
          type="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="input"
        />
      </div>

      {error ? (
        <p role="alert" className="text-sm text-red-600">
          {error.message}
        </p>
      ) : null}

      <button type="submit" disabled={loading} className="btn-primary">
        {loading ? <LoadingSpinner size="sm" /> : null}
        Sign in
      </button>

      <p className="text-sm text-muted-foreground">
        No account?{" "}
        <Link href="/register" className="text-brand underline-offset-4 hover:underline">
          Create one
        </Link>
      </p>
    </form>
  );
}
