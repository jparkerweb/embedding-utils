# AGENTS.md

This file provides guidance to AI coding agents like Claude Code (claude.ai/code), Cursor AI, Codex, Gemini CLI, GitHub Copilot, Devin, and other AI coding assistants when working with code in this repository.

## Project Overview

`embedding-utils` is a lightweight, **zero-production-dependency**, provider-agnostic TypeScript library for working with text embeddings in Node.js (>=18). It bundles vector math, similarity & ANN search (HNSW), clustering (k-means presets, HDBSCAN), aggregation, quantization, dimensionality reduction, retrieval evaluation metrics, async embedding pipelines, caching/serialization, markdown-aware chunking, and multi-provider embedding generation (local ONNX, OpenAI-compatible, Cohere, Google Vertex) behind a single import.

It ships as a dual **ESM + CJS** build with full type declarations, is tree-shakeable (`sideEffects: false`), and treats `@huggingface/transformers` as an **optional** peer dependency (only needed for the local ONNX provider). The public surface is a flat set of named exports plus grouped namespace objects (`Math`, `Search`, `Clustering`, `Eval`, `Pipeline`), all re-exported from `src/index.ts`.

## How to Use This File

The sections below are brief summaries. Each links to a detail file in `.agents-docs/` with the full content. **Follow the markdown links only for the area you are working in** — don't read everything up front. Read the relevant detail file before making changes to that subsystem.

## Development Commands

`npm run build` (tsup → dual ESM/CJS + dts), `npm test` (vitest run), `npm run test:watch`, `npm run test:coverage`, `npm run lint` (eslint over `src/` + `tests/`), `npm run format` / `npm run format:check` (prettier), `npm run smoke` (live ONNX, local-only), `npm run engines:check`. Tests live in `tests/` mirroring `src/`. CI (`.github/workflows/ci.yml`) enforces all of these except `smoke`; `package-lock.json` is tracked, so use `npm ci`.

Three temporary `overrides` in `package.json` (`sharp`, `adm-zip`, `esbuild`) hold transitive dependencies at non-vulnerable versions, and TypeScript is held at 6.x. Both have removal conditions documented — read the detail file before changing either.

Details: [Development Commands](./.agents-docs/AGENTS-development-commands.md)

## Architecture

Source is organized into self-contained feature modules under `src/`, each with an `index.ts` barrel re-exported by the root `src/index.ts`. Shared, non-public helpers live in `src/internal/`; all public types live in `src/types.ts`. Providers implement a common `EmbeddingProvider` interface so they're interchangeable.

Details: [Architecture](./.agents-docs/AGENTS-architecture.md)

## Code Conventions

Strict TypeScript, ESM-only source, Prettier (single quotes, semicolons, 100-col, es5 trailing commas). Embeddings are `Float32Array` (v0.3+). New public exports must flow through the module barrel and `src/index.ts`; internal helpers stay in `src/internal/` and are `@internal`.

Details: [Code Conventions](./.agents-docs/AGENTS-code-conventions.md)
