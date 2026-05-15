export type AuthSessionLike = {
  key: string;
};

export function createAuthSessionValidationMemo<TSession extends AuthSessionLike>(
  ttlMs: number,
  now: () => number = () => Date.now(),
) {
  let cachedSession: TSession | null = null;
  let cachedAt = 0;
  let inFlightKey = "";
  let inFlight: Promise<TSession | null> | null = null;

  const clear = () => {
    cachedSession = null;
    cachedAt = 0;
    inFlightKey = "";
    inFlight = null;
  };

  const prime = (session: TSession, validatedAt = now()) => {
    cachedSession = session;
    cachedAt = validatedAt;
  };

  const validate = async (
    storedSession: TSession | null,
    refresh: (storedSession: TSession) => Promise<TSession | null>,
  ): Promise<TSession | null> => {
    if (!storedSession) {
      clear();
      return null;
    }

    if (
      cachedSession?.key === storedSession.key &&
      now() - cachedAt < ttlMs
    ) {
      return cachedSession;
    }

    if (inFlight && inFlightKey === storedSession.key) {
      return inFlight;
    }

    inFlightKey = storedSession.key;
    inFlight = refresh(storedSession)
      .then((nextSession) => {
        if (nextSession?.key === storedSession.key) {
          prime(nextSession);
          return nextSession;
        }
        cachedSession = null;
        cachedAt = 0;
        return nextSession;
      })
      .finally(() => {
        inFlight = null;
        inFlightKey = "";
      });

    return inFlight;
  };

  return { clear, prime, validate };
}
