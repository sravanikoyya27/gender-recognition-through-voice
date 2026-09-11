"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";

import LoadingSpinner from "@/components/LoadingSpinner";
import { useAuth } from "@/hooks/useAuth";

export default function RegisterPage() {
  const router = useRouter();
  const { register, loading, error } = useAuth();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const ok = await register({ email, password, full_name: fullName || undefined });
    if (ok) router.push("/login");
  }

  return (
    <form onSubmit={handleSubmit} className="card flex flex-col gap-4">
      <h1 className="text-h2">Create an account</h1>

      <div>
        <label htmlFor="fullName" className="label">
          Full name <span className="text-muted-foreground">(optional)</span>
        </label>
        <input
          id="fullName"
          type="text"
          autoComplete="name"
          value={fullName}
          onChange={(e) => setFullName(e.target.value)}
          className="input"
        />
      </div>

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
          autoComplete="new-password"
          required
          minLength={8}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="input"
        />
        <p className="mt-1 text-xs text-muted-foreground">At least 8 characters.</p>
      </div>

      {error ? (
        <p role="alert" className="text-sm text-red-600">
          {error.message}
        </p>
      ) : null}

      <button type="submit" disabled={loading} className="btn-primary">
        {loading ? <LoadingSpinner size="sm" /> : null}
        Create account
      </button>

      <p className="text-sm text-muted-foreground">
        Already registered?{" "}
        <Link href="/login" className="text-brand underline-offset-4 hover:underline">
          Sign in
        </Link>
      </p>
    </form>
  );
}
