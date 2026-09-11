"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";

import LoadingSpinner from "@/components/LoadingSpinner";
import { useAuth } from "@/hooks/useAuth";

export default function LoginPage() {
  const router = useRouter();
  const { login, loading, error } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const ok = await login({ email, password });
    if (ok) router.push("/dashboard");
  }

  return (
    <form onSubmit={handleSubmit} className="card flex flex-col gap-4">
      <h1 className="text-h2">Sign in</h1>

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
