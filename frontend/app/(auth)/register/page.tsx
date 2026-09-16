"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";

import LoadingSpinner from "@/components/LoadingSpinner";
import { useAuth } from "@/hooks/useAuth";
import { PASSWORD_MAX, PASSWORD_MIN, type RegisterErrors, validateRegister } from "@/lib/authValidation";
import { cn } from "@/lib/utils";
import type { RegisterRequest } from "@/types/api";

const EMPTY: RegisterRequest = { username: "", email: "", password: "", password_confirm: "" };

export default function RegisterPage() {
  const router = useRouter();
  const { register, loading, error } = useAuth();
  const [values, setValues] = useState<RegisterRequest>(EMPTY);
  const [fieldErrors, setFieldErrors] = useState<RegisterErrors>({});

  function update(field: keyof RegisterRequest, value: string) {
    setValues((v) => ({ ...v, [field]: value }));
    if (fieldErrors[field]) setFieldErrors((e) => ({ ...e, [field]: undefined }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const payload: RegisterRequest = { ...values, email: values.email.trim() };
    const errors = validateRegister(payload);
    setFieldErrors(errors);
    if (Object.keys(errors).length > 0) return;
    const created = await register(payload);
    if (created) router.push(`/login?registered=${encodeURIComponent(created.username)}`);
  }

  const field = (
    name: keyof RegisterRequest,
    label: string,
    type: string,
    autoComplete: string,
    hint?: string,
  ) => (
    <div>
      <label htmlFor={name} className="label">
        {label}
      </label>
      <input
        id={name}
        name={name}
        type={type}
        autoComplete={autoComplete}
        required
        value={values[name]}
        onChange={(e) => update(name, e.target.value)}
        aria-invalid={fieldErrors[name] ? true : undefined}
        aria-describedby={fieldErrors[name] ? `${name}-error` : hint ? `${name}-hint` : undefined}
        className={cn("input", fieldErrors[name] && "border-red-500")}
      />
      {fieldErrors[name] ? (
        <p id={`${name}-error`} role="alert" className="mt-1 text-xs text-red-600">
          {fieldErrors[name]}
        </p>
      ) : hint ? (
        <p id={`${name}-hint`} className="mt-1 text-xs text-muted-foreground">
          {hint}
        </p>
      ) : null}
    </div>
  );

  return (
    <form onSubmit={handleSubmit} noValidate className="card flex flex-col gap-4">
      <h1 className="text-h2">Create an account</h1>

      {field("username", "Username", "text", "username", "3-50 letters or digits.")}
      {field("email", "Email", "email", "email")}
      {field(
        "password",
        "Password",
        "password",
        "new-password",
        `${PASSWORD_MIN} to ${PASSWORD_MAX} characters.`,
      )}
      {field("password_confirm", "Confirm password", "password", "new-password")}

      {error ? (
        <p role="alert" className="text-sm text-red-600">
          {error.status === 409
            ? `${error.message}. Choose another or sign in.`
            : error.message}
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
