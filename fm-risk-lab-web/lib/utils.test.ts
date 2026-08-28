/**
 * Regression tests for lib/utils.ts.
 *
 * Run with: node --experimental-strip-types --test lib/utils.test.ts
 * (Node's built-in test runner + type stripping -- no extra dev dependency
 * needed for a single pure-function test file.)
 */
import test from "node:test";
import assert from "node:assert/strict";

import { scenarioDeltaMessage } from "./utils.ts";

test("baseline (delta_pct === 0) is labeled baseline", () => {
  const msg = scenarioDeltaMessage(0);
  assert.equal(msg.tone, "baseline");
});

test("mild stress (0 < delta_pct < 30) is labeled mild", () => {
  const msg = scenarioDeltaMessage(1.563);
  assert.equal(msg.tone, "mild");
  assert.match(msg.text, /1\.6%/);
});

test("severe stress (delta_pct >= 30) is labeled severe, boundary included", () => {
  assert.equal(scenarioDeltaMessage(30).tone, "severe");
  assert.equal(scenarioDeltaMessage(45).tone, "severe");
});

test("regression: a favorable scenario (delta_pct < 0) is not left unhandled", () => {
  // These are real ecl_delta_pct values the backend returns for hpi_shock
  // alone at -1, -5, and -40 (see src/models/ecl_engine.py ECLEngine /
  // /tmp verification), i.e. the ordinary case of a user moving only the
  // HPI slider negative on the risk-lab page. Before this fix, the
  // three-branch version of this logic (==0, (0,30), >=30) matched none of
  // these and the UI rendered a blank micro-copy card.
  for (const deltaPct of [-0.862, -3.961, -8.799]) {
    const msg = scenarioDeltaMessage(deltaPct);
    assert.equal(msg.tone, "improved", `delta_pct=${deltaPct} should be tone "improved"`);
    assert.ok(msg.text.length > 0, `delta_pct=${deltaPct} must render non-empty text`);
    assert.doesNotMatch(msg.text, /^$/);
  }
});

test("every finite delta_pct produces exactly one non-empty message", () => {
  for (const deltaPct of [-100, -40, -0.001, 0, 0.001, 15, 29.999, 30, 30.001, 100]) {
    const msg = scenarioDeltaMessage(deltaPct);
    assert.ok(["baseline", "improved", "mild", "severe"].includes(msg.tone));
    assert.ok(msg.text.length > 0);
  }
});
