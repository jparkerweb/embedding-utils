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

## Dependency overrides

`package.json` declares `overrides` to pull two transitive dependencies of `@huggingface/transformers` above the versions it pins, resolving high-severity advisories that have no upstream fix:

| Override | Pinned by transformers | Advisory |
|---|---|---|
| `sharp: ^0.35.3` | `^0.34.5` | libvips CVEs (GHSA-f88m-g3jw-g9cj) |
| `adm-zip: ^0.6.0` | `0.5.x` via `onnxruntime-node@1.24.3` | GHSA-xcpc-8h2w-3j85 |

Both are outside the ranges transformers declares, so **re-run `npm run smoke` after changing them** — it exercises `onnxruntime-node` end-to-end across three models and is the only guard that these overrides remain safe.

`esbuild` is also pinned to exactly `0.27.2`. tsup declares `esbuild: ^0.27.0`, and every version from `0.27.3` up to `0.28.0` is covered by GHSA-g7r4-m6w7-qqqr, so `0.27.2` is the only in-range version without the advisory. The override (rather than a lockfile pin) is required because `package-lock.json` is gitignored — a fresh `npm install` would otherwise resolve a vulnerable `0.27.x`. Remove the pin once tsup widens its range to `^0.28.1` or later.

## TypeScript version constraints

- The project is on **TypeScript 6.0.x**. Do not bump to TypeScript 7 yet: `@typescript-eslint/*` v8 declares `typescript: ">=4.8.4 <6.1.0"`, so `npm run lint` would run against an unsupported compiler.
- `tsconfig.json` sets `"ignoreDeprecations": "6.0"`. This is **required for `npm run build`**, not for our own config — tsup's dts pipeline hardcodes `baseUrl: compilerOptions.baseUrl || "."` (`node_modules/tsup/dist/rollup.js`), and TS 6 raises `TS5101` for deprecated `baseUrl`. Removing the flag breaks the DTS build. It can be dropped once tsup stops injecting `baseUrl`.
