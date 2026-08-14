// Verifies the built dist/ actually runs on the oldest Node major that
// package.json's `engines.node` claims to support.
//
// This exists because the dev toolchain (vitest, eslint) requires Node >=20,
// so the CI matrix cannot exercise Node 18 the normal way. Only the published
// artifact needs to work there, and it has no runtime dependencies - so this
// script imports dist/ directly and needs nothing installed.
//
// Run with: node scripts/engines-check.mjs

import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const require = createRequire(import.meta.url);
const { engines, version } = JSON.parse(readFileSync('./package.json', 'utf8'));

const a = [1, 0, 0];
const b = [0.6, 0.8, 0];

function check(label, mod) {
  assert.equal(typeof mod.cosineSimilarity, 'function', `${label}: cosineSimilarity missing`);
  assert.equal(typeof mod.dotProduct, 'function', `${label}: dotProduct missing`);
  assert.equal(typeof mod.topK, 'function', `${label}: topK missing`);

  // 1*0.6 + 0*0.8 + 0*0 = 0.6, and both inputs are unit vectors.
  assert.ok(Math.abs(mod.cosineSimilarity(a, b) - 0.6) < 1e-6, `${label}: cosineSimilarity wrong`);
  assert.ok(Math.abs(mod.dotProduct(a, b) - 0.6) < 1e-6, `${label}: dotProduct wrong`);

  const ranked = mod.topK(a, [b, [1, 0, 0], [0, 1, 0]], 2);
  assert.equal(ranked.length, 2, `${label}: topK returned ${ranked.length} results, expected 2`);
  assert.equal(ranked[0].index, 1, `${label}: topK did not rank the identical vector first`);

  console.log(`    ok  ${label}`);
}

console.log(`embedding-utils ${version} on Node ${process.version} (engines: ${engines.node})`);

check('esm  (dist/index.js)', await import('../dist/index.js'));
check('cjs  (dist/index.cjs)', require('../dist/index.cjs'));

console.log('All engines checks passed.');
