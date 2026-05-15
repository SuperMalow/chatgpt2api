"use client";

import { login } from "@/lib/api";
import { createAuthSessionValidationMemo } from "@/lib/auth-session-validation";
import { clearStoredAuthSession, getStoredAuthSession, setStoredAuthSession, type StoredAuthSession } from "@/store/auth";

const authSessionValidation = createAuthSessionValidationMemo<StoredAuthSession>(60_000);

export async function getValidatedAuthSession(): Promise<StoredAuthSession | null> {
  const storedSession = await getStoredAuthSession();
  if (!storedSession) {
    authSessionValidation.clear();
    return null;
  }

  try {
    const nextSession = await authSessionValidation.validate(storedSession, async (session) => {
      const data = await login(session.key);
      return {
        key: session.key,
        role: data.role,
        subjectId: data.subject_id,
        name: data.name,
      };
    });

    if (!nextSession) {
      await clearStoredAuthSession();
      return null;
    }

    await setStoredAuthSession(nextSession);
    return nextSession;
  } catch {
    authSessionValidation.clear();
    await clearStoredAuthSession();
    return null;
  }
}

export function primeValidatedAuthSession(session: StoredAuthSession) {
  authSessionValidation.prime(session);
}

export function getValidatedAuthSessionSnapshot() {
  return authSessionValidation.snapshot();
}

export function clearValidatedAuthSessionCache() {
  authSessionValidation.clear();
}
