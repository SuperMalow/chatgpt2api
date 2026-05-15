"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { getValidatedAuthSession, getValidatedAuthSessionSnapshot } from "@/lib/auth-session";
import {
  getDefaultRouteForRole,
  type AuthRole,
  type StoredAuthSession,
} from "@/store/auth";

type UseAuthGuardResult = {
  isCheckingAuth: boolean;
  session: StoredAuthSession | null;
};

function parseAllowedRoles(allowedRolesKey: string) {
  return allowedRolesKey ? (allowedRolesKey.split(",") as AuthRole[]) : [];
}

function canUseSession(session: StoredAuthSession | null, allowedRolesKey: string) {
  if (!session) {
    return false;
  }
  const roleList = parseAllowedRoles(allowedRolesKey);
  return roleList.length === 0 || roleList.includes(session.role);
}

export function useAuthGuard(allowedRoles?: AuthRole[]): UseAuthGuardResult {
  const router = useRouter();
  const allowedRolesKey = (allowedRoles || []).join(",");
  const [session, setSession] = useState<StoredAuthSession | null>(() => getValidatedAuthSessionSnapshot());
  const [isCheckingAuth, setIsCheckingAuth] = useState(() => {
    const cachedSession = getValidatedAuthSessionSnapshot();
    return !canUseSession(cachedSession, allowedRolesKey);
  });

  useEffect(() => {
    let active = true;

    const load = async () => {
      const roleList = parseAllowedRoles(allowedRolesKey);
      const storedSession = await getValidatedAuthSession();
      if (!active) {
        return;
      }

      if (!storedSession) {
        setSession(null);
        setIsCheckingAuth(false);
        router.replace("/login");
        return;
      }

      if (roleList.length > 0 && !roleList.includes(storedSession.role)) {
        setSession(storedSession);
        setIsCheckingAuth(false);
        router.replace(getDefaultRouteForRole(storedSession.role));
        return;
      }

      setSession(storedSession);
      setIsCheckingAuth(false);
    };

    void load();
    return () => {
      active = false;
    };
  }, [allowedRolesKey, router]);

  return { isCheckingAuth, session };
}

export function useRedirectIfAuthenticated() {
  const router = useRouter();
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

  useEffect(() => {
    let active = true;

    const load = async () => {
      const storedSession = await getValidatedAuthSession();
      if (!active) {
        return;
      }

      if (storedSession) {
        router.replace(getDefaultRouteForRole(storedSession.role));
        return;
      }

      setIsCheckingAuth(false);
    };

    void load();
    return () => {
      active = false;
    };
  }, [router]);

  return { isCheckingAuth };
}
