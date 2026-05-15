import assert from "node:assert/strict";
import test from "node:test";

import { createAuthSessionValidationMemo } from "../src/lib/auth-session-validation.ts";

type TestSession = {
  key: string;
  role: "admin" | "user";
  subjectId: string;
  name: string;
};

const session: TestSession = {
  key: "secret-key",
  role: "admin",
  subjectId: "admin",
  name: "Admin",
};

test("deduplicates concurrent auth session validation requests", async () => {
  let now = 0;
  let calls = 0;
  const memo = createAuthSessionValidationMemo<TestSession>(60_000, () => now);

  const refresh = async (storedSession: TestSession) => {
    calls += 1;
    await Promise.resolve();
    return { ...storedSession, name: "Validated Admin" };
  };

  const results = await Promise.all([
    memo.validate(session, refresh),
    memo.validate(session, refresh),
    memo.validate(session, refresh),
  ]);

  assert.equal(calls, 1);
  assert.deepEqual(results.map((item) => item?.name), [
    "Validated Admin",
    "Validated Admin",
    "Validated Admin",
  ]);

  await memo.validate(session, refresh);
  assert.equal(calls, 1);

  now = 60_001;
  await memo.validate(session, refresh);
  assert.equal(calls, 2);
});

test("does not reuse validation cache for a different auth key", async () => {
  let calls = 0;
  const memo = createAuthSessionValidationMemo<TestSession>(60_000, () => 0);

  const refresh = async (storedSession: TestSession) => {
    calls += 1;
    return storedSession;
  };

  await memo.validate(session, refresh);
  await memo.validate({ ...session, key: "other-key" }, refresh);

  assert.equal(calls, 2);
});
