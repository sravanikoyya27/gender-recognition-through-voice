/** Client-side checks mirroring backend/app/schemas/user.py (UserRegister / UserLogin). */

import type { RegisterRequest } from "@/types/api";

export const USERNAME_PATTERN = /^[A-Za-z0-9]{3,50}$/;
export const PASSWORD_MIN = 8;
export const PASSWORD_MAX = 128;
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export type RegisterErrors = Partial<Record<keyof RegisterRequest, string>>;

export function validateRegister(values: RegisterRequest): RegisterErrors {
  const errors: RegisterErrors = {};
  if (!USERNAME_PATTERN.test(values.username)) {
    errors.username = "3-50 letters or digits, no spaces or symbols.";
  }
  if (!EMAIL_PATTERN.test(values.email.trim())) {
    errors.email = "Enter a valid email address.";
  }
  if (values.password.length < PASSWORD_MIN || values.password.length > PASSWORD_MAX) {
    errors.password = `Between ${PASSWORD_MIN} and ${PASSWORD_MAX} characters.`;
  }
  if (values.password_confirm !== values.password) {
    errors.password_confirm = "Passwords do not match.";
  }
  return errors;
}

/** Only same-origin paths are safe redirect targets after sign-in. */
export function safeNextPath(value: string | null, fallback = "/dashboard"): string {
  return value && value.startsWith("/") && !value.startsWith("//") ? value : fallback;
}
