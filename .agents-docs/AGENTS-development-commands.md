# Development Commands
> Part of [AGENTS.md](../AGENTS.md) — project guidance for AI coding agents.

## Scripts (`package.json`)

| Command | What it does |
|---------|--------------|
| `npm run build` | Bundles `src/index.ts` with **tsup** → ESM (`dist/index.js`) + CJS (`dist/index.cjs`) + type declarations (`.d.ts` / `.d.cts`). Config: `tsup.config.ts` (target `es2022`, `splitting`, `sourcemap`, `clean`). |
| `npm test` | Runs the full Vitest suite once (`vitest run`). |
| `npm run test:watch` | Vitest in watch mode. |
| `npm run test:coverage` | Vitest with v8 coverage (`text` + `lcov`). |
| `npm run lint` | ESLint over `src/`. |
| `npm run format` | Prettier write over `src/**/*.ts` and `tests/**/*.ts`. |

## Running a single test

Vitest passes through file/name filters:

```bash
npx vitest run tests/search/mmr.test.ts          # one file
npx vitest run -t "mmrSearch returns diverse"      # by test name
npx vitest tests/clustering                         # watch a directory
```

## Test layout

- Tests live in `tests/`, mirroring the `src/` module structure (e.g. `src/search/mmr.ts` → `tests/search/mmr.test.ts`).
- Vitest config (`vitest.config.ts`) includes `tests/**/*.test.ts`; coverage includes `src/**` but excludes `src/index.ts` (pure re-export barrel).
- Integration tests live under `tests/integration/`.

## Build/release notes

- `dist/` is the published artifact; `files` in `package.json` ships `dist`, `README.md`, `LICENSE`.
- `@huggingface/transformers` is an **optional peer dependency** — present in `devDependencies` for local-provider tests but never required for consumers who don't use the local ONNX provider.
- Engines: Node `>=18`.
