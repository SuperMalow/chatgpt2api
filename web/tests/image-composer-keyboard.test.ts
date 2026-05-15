import assert from "node:assert/strict";
import test from "node:test";

import { shouldSubmitImageComposerPrompt } from "../src/app/image/components/image-composer-keyboard.ts";

test("plain Enter submits the prompt", () => {
  assert.equal(
    shouldSubmitImageComposerPrompt({
      key: "Enter",
      shiftKey: false,
      nativeEvent: { isComposing: false },
    }),
    true,
  );
});

test("Shift+Enter keeps a newline", () => {
  assert.equal(
    shouldSubmitImageComposerPrompt({
      key: "Enter",
      shiftKey: true,
      nativeEvent: { isComposing: false },
    }),
    false,
  );
});

test("Enter during IME composition does not submit", () => {
  assert.equal(
    shouldSubmitImageComposerPrompt({
      key: "Enter",
      shiftKey: false,
      nativeEvent: { isComposing: true },
    }),
    false,
  );
});

test("IME process key events do not submit on browsers using keyCode 229", () => {
  assert.equal(
    shouldSubmitImageComposerPrompt({
      key: "Enter",
      shiftKey: false,
      nativeEvent: { keyCode: 229 },
    }),
    false,
  );

  assert.equal(
    shouldSubmitImageComposerPrompt({
      key: "Enter",
      shiftKey: false,
      nativeEvent: { which: 229 },
    }),
    false,
  );
});
