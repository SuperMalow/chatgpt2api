"use client";

export type AuthRole = "admin" | "user";

export type StoredAuthSession = {
  key: string;
  role: AuthRole;
  subjectId: string;
  name: string;
};

export const AUTH_KEY_STORAGE_KEY = "chatgpt2api_auth_key";
export const AUTH_SESSION_STORAGE_KEY = "chatgpt2api_auth_session";
const AUTH_LEGACY_IGNORED_KEY = "chatgpt2api_auth_legacy_ignored";

let legacyMigrationPromise: Promise<StoredAuthSession | null> | null = null;

function normalizeSession(value: unknown, fallbackKey = ""): StoredAuthSession | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const candidate = value as Partial<StoredAuthSession>;
  const key = String(candidate.key || fallbackKey || "").trim();
  const role = candidate.role === "admin" || candidate.role === "user" ? candidate.role : null;
  if (!key || !role) {
    return null;
  }

  return {
    key,
    role,
    subjectId: String(candidate.subjectId || "").trim(),
    name: String(candidate.name || "").trim(),
  };
}

export function getDefaultRouteForRole(role: AuthRole) {
  return role === "admin" ? "/accounts" : "/image";
}

function getLocalStorage() {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function readStoredSessionFromLocalStorage() {
  const storage = getLocalStorage();
  if (!storage) {
    return null;
  }

  const storedKey = String(storage.getItem(AUTH_KEY_STORAGE_KEY) || "").trim();
  const rawSession = storage.getItem(AUTH_SESSION_STORAGE_KEY);
  let storedSession: unknown = null;
  if (rawSession) {
    try {
      storedSession = JSON.parse(rawSession);
    } catch {
      storedSession = null;
    }
  }

  return normalizeSession(storedSession, storedKey);
}

async function migrateLegacyAuthSession() {
  const storage = getLocalStorage();
  if (!storage || storage.getItem(AUTH_LEGACY_IGNORED_KEY)) {
    return null;
  }
  if (!legacyMigrationPromise) {
    legacyMigrationPromise = (async () => {
      try {
        const localforage = (await import("localforage")).default;
        const authStorage = localforage.createInstance({
          name: "chatgpt2api",
          storeName: "auth",
        });
        const [storedKey, storedSession] = await Promise.all([
          authStorage.getItem<string>(AUTH_KEY_STORAGE_KEY),
          authStorage.getItem<StoredAuthSession>(AUTH_SESSION_STORAGE_KEY),
        ]);
        const normalizedSession = normalizeSession(storedSession, String(storedKey || ""));
        if (normalizedSession) {
          storage.setItem(AUTH_KEY_STORAGE_KEY, normalizedSession.key);
          storage.setItem(AUTH_SESSION_STORAGE_KEY, JSON.stringify(normalizedSession));
        }
        return normalizedSession;
      } catch {
        return null;
      }
    })();
  }
  return legacyMigrationPromise;
}

export async function getStoredAuthKey() {
  const storage = getLocalStorage();
  const value = String(storage?.getItem(AUTH_KEY_STORAGE_KEY) || "").trim();
  if (value) {
    return value;
  }
  const legacySession = await migrateLegacyAuthSession();
  return legacySession?.key || "";
}

export async function getStoredAuthSession() {
  const storage = getLocalStorage();
  if (!storage) {
    return null;
  }

  const normalizedSession = readStoredSessionFromLocalStorage();
  if (normalizedSession) {
    return normalizedSession;
  }

  const storedKey = String(storage.getItem(AUTH_KEY_STORAGE_KEY) || "").trim();
  if (!storedKey) {
    return migrateLegacyAuthSession();
  }
  if (storedKey) {
    await clearStoredAuthSession();
  }
  return null;
}

export async function setStoredAuthSession(session: StoredAuthSession) {
  const normalizedSession = normalizeSession(session);
  if (!normalizedSession) {
    await clearStoredAuthSession();
    return;
  }

  const storage = getLocalStorage();
  if (!storage) {
    return;
  }
  storage.removeItem(AUTH_LEGACY_IGNORED_KEY);
  storage.setItem(AUTH_KEY_STORAGE_KEY, normalizedSession.key);
  storage.setItem(AUTH_SESSION_STORAGE_KEY, JSON.stringify(normalizedSession));
}

export async function setStoredAuthKey(authKey: string) {
  const normalizedAuthKey = String(authKey || "").trim();
  if (!normalizedAuthKey) {
    await clearStoredAuthSession();
    return;
  }
  const storage = getLocalStorage();
  if (!storage) {
    return;
  }
  storage.removeItem(AUTH_LEGACY_IGNORED_KEY);
  storage.setItem(AUTH_KEY_STORAGE_KEY, normalizedAuthKey);
  storage.removeItem(AUTH_SESSION_STORAGE_KEY);
}

export async function clearStoredAuthSession() {
  const storage = getLocalStorage();
  if (!storage) {
    return;
  }
  storage.removeItem(AUTH_KEY_STORAGE_KEY);
  storage.removeItem(AUTH_SESSION_STORAGE_KEY);
  storage.setItem(AUTH_LEGACY_IGNORED_KEY, String(Date.now()));
}

export async function clearStoredAuthKey() {
  await clearStoredAuthSession();
}
