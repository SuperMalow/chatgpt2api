import assert from "node:assert/strict";
import test from "node:test";

import {
  getImageTurnAspectClass,
  getImageTurnAspectRatio,
} from "../src/app/image/components/image-results-layout.ts";

test("maps image sizes to stable aspect ratio classes", () => {
  assert.equal(getImageTurnAspectClass("1:1"), "aspect-square");
  assert.equal(getImageTurnAspectClass("16:9"), "aspect-video");
  assert.equal(getImageTurnAspectClass("9:16"), "aspect-[9/16]");
  assert.equal(getImageTurnAspectClass("4:3"), "aspect-[4/3]");
  assert.equal(getImageTurnAspectClass("3:4"), "aspect-[3/4]");
});

test("keeps mobile square crop only for result placeholders that request it", () => {
  assert.equal(
    getImageTurnAspectClass("16:9", { mobileSquare: true }),
    "aspect-square sm:aspect-video",
  );
  assert.equal(
    getImageTurnAspectClass("9:16", { mobileSquare: true }),
    "aspect-square sm:aspect-[9/16]",
  );
});

test("uses explicit aspect ratios for loading skeleton dimensions", () => {
  assert.equal(getImageTurnAspectRatio("16:9"), "16 / 9");
  assert.equal(getImageTurnAspectRatio("9:16"), "9 / 16");
  assert.equal(getImageTurnAspectRatio("unknown"), "1 / 1");
});
