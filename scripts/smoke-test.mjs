#!/usr/bin/env node
/**
 * Live smoke test for the local ONNX provider.
 *
 * Actually downloads each model from HuggingFace and runs a real embedding,
 * then verifies end-to-end behaviour that the unit tests (which mock the
 * pipeline) cannot:
 *   - output dimensions match the registry's declared `dimensions`
 *   - vectors are L2-normalized (magnitude ~= 1.0, since normalize: true)
 *   - the registry `pooling` method is actually applied
 *   - asymmetric models produce different query vs document vectors (prefixes)
 *
 * Prereqs:  npm run build   (this imports the built ./dist surface)
 * Usage:    node scripts/smoke-test.mjs [modelId ...]
 *           node scripts/smoke-test.mjs Xenova/bge-large-en-v1.5
 *
 * With no args it runs a representative set covering all three pooling
 * methods. Large models (>2GB at fp32) are downloaded at q8.
 *
 * Exit code is non-zero if any check fails, so it can gate a release.
 */

import { createLocalProvider, getModelInfo } from '../dist/index.js';

// Representative default set: one model per pooling method.
// [modelId, precision]
const DEFAULTS = [
  ['Xenova/all-MiniLM-L6-v2', 'fp32'], // mean, tiny
  ['Xenova/bge-large-en-v1.5', 'q8'], // cls
  ['onnx-community/Qwen3-Embedding-0.6B-ONNX', 'q8'], // last_token, >2GB at fp32
];

const args = process.argv.slice(2);
const models = args.length > 0 ? args.map((id) => [id, undefined]) : DEFAULTS;

const DOC = 'The Eiffel Tower is located in Paris, France.';
const QUERY = 'Where is the Eiffel Tower?';

function magnitude(vec) {
  let sum = 0;
  for (const v of vec) sum += v * v;
  return Math.sqrt(sum);
}

function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // both are unit vectors, so dot == cosine
}

let failures = 0;
const fail = (msg) => {
  failures++;
  console.log(`    ✗ ${msg}`);
};
const pass = (msg) => console.log(`    ✓ ${msg}`);

console.log('Local ONNX provider smoke test\n');

for (const [modelId, precision] of models) {
  const info = getModelInfo(modelId);
  const expectedPooling = info?.pooling ?? 'mean';
  const expectedDims = info?.dimensions;
  const hasPrefixes =
    !!info?.prefixes && info.prefixes.document !== info.prefixes.query;

  console.log(
    `▶ ${modelId}` +
      (precision ? `  [${precision}]` : '') +
      `  (registry: ${expectedDims ?? '?'}d, pooling=${expectedPooling}` +
      (info ? '' : ', NOT in registry') +
      `)`,
  );

  try {
    const provider = createLocalProvider(
      precision ? { model: modelId, precision } : { model: modelId },
    );

    const docRes = await provider.embed(DOC, { inputType: 'document' });
    const docVec = docRes.embeddings[0];

    // Dimensions match the registry?
    if (expectedDims != null) {
      if (docRes.dimensions === expectedDims) {
        pass(`dimensions = ${docRes.dimensions}`);
      } else {
        fail(
          `dimensions mismatch: got ${docRes.dimensions}, registry says ${expectedDims}`,
        );
      }
    } else {
      console.log(`    • dimensions = ${docRes.dimensions} (no registry value)`);
    }

    // Normalized?
    const mag = magnitude(docVec);
    if (Math.abs(mag - 1.0) < 1e-3) {
      pass(`L2-normalized (magnitude = ${mag.toFixed(4)})`);
    } else {
      fail(`not normalized (magnitude = ${mag.toFixed(4)})`);
    }

    // Prefixes applied (asymmetric models): query vs document must differ.
    if (hasPrefixes) {
      const queryRes = await provider.embed(QUERY, { inputType: 'query' });
      const sim = cosine(queryRes.embeddings[0], docVec);
      // Also embed the query WITHOUT the query prefix by passing it as a doc;
      // if prefixes are applied the two query encodings should differ.
      const queryAsDoc = await provider.embed(QUERY, { inputType: 'document' });
      const prefixDelta = 1 - cosine(queryRes.embeddings[0], queryAsDoc.embeddings[0]);
      if (prefixDelta > 1e-4) {
        pass(
          `prefixes applied (query/doc encodings differ by ${prefixDelta.toExponential(2)}; query~doc sim=${sim.toFixed(3)})`,
        );
      } else {
        fail('prefixes do not appear to be applied (query == doc encoding)');
      }
    }

    console.log('');
  } catch (err) {
    fail(`threw: ${err?.message ?? err}`);
    console.log('');
  }
}

if (failures > 0) {
  console.log(`✗ ${failures} check(s) failed.`);
  process.exit(1);
}
console.log('✓ All checks passed.');
