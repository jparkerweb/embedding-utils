# Development Commands
> Part of [AGENTS.md](../AGENTS.md) — project guidance for AI coding agents.

## Scripts (`package.json`)

| Command | What it does |
|---------|--------------|
| `npm run build` | Bundles `src/index.ts` with **tsup** → ESM (`dist/index.js`) + CJS (`dist/index.cjs`) + type declarations (`.d.ts` / `.d.cts`). Config: `tsup.config.ts` (target `es2022`, `splitting`, `sourcemap`, `clean`). |
| `npm test` | Runs the full Vitest suite once (`vitest run`). |
| `npm run test:watch` | Vitest in watch mode. |
| `npm run test:coverage` | Vitest with v8 coverage (`text` + `lcov`). |
| `npm run lint` | ESLint over `src/` **and** `tests/`. |
| `npm run format` | Prettier write over `src/**/*.ts` and `tests/**/*.ts`. |
| `npm run format:check` | Prettier check (no writes) — used in CI. |
| `npm run smoke` | Builds, then runs the live local-ONNX provider test (`scripts/smoke-test.mjs`). Downloads models from HuggingFace, so it is local-only and not run in CI. |
| `npm run engines:check` | Runs the built `dist/` (ESM + CJS) and asserts real behaviour, to verify the `engines.node` floor. See CI below. |

Glob arguments in the `format` scripts must use **double** quotes. Single quotes are not stripped by `cmd`/PowerShell, so prettier receives the literal glob, matches nothing, and the script silently succeeds as a no-op on Windows.

`eslint.config.mjs` relaxes two rules for `tests/**` only: `no-explicit-any` is off (mocking `fetch`/`setTimeout`/provider responses legitimately needs `any`) and `no-unused-vars` honours the `^_` prefix for intentionally unused arguments. Neither relaxation applies to `src/`. Scripts under `scripts/` are `.mjs` and are covered by neither tool.

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
- Engines: Node `>=18`. The dev toolchain itself needs `>=20`, so 18 is verified against the built artifact only — see CI below.
- `package-lock.json` **is tracked**. Use `npm ci` for reproducible installs; do not re-ignore it.

## CI

`.github/workflows/ci.yml` runs on pushes to `master`/`develop`, on every pull request, and weekly on a schedule.

| Job | What it does |
|---|---|
| `verify` | `format:check`, `lint`, `typecheck`, `test`, `build` across Node 20, 22, 24. |
| `engines-floor` | Builds on Node 24, then runs `npm run engines:check` on Node 18. |
| `audit` | `npm audit --audit-level=high` against the committed lockfile. |

Notes for anyone editing the workflow:

- The `verify` matrix floor is **20, not 18**, because vitest 4 requires `^20 \|\| ^22 \|\| >=24` and eslint 10 requires `^20.19 \|\| ^22.13 \|\| >=24`. Node 18 cannot run the toolchain at all. `engines-floor` covers the `>=18` claim instead by exercising the built `dist/`, which has no runtime dependencies. If `engines.node` changes, update that job.
- Installs use `npm ci --ignore-scripts` to skip the `onnxruntime-node` and `sharp` postinstalls (hundreds of MB of native binaries). Nothing in CI needs them. If a future test does need real ONNX, drop `--ignore-scripts` for that job rather than adding it to `verify`.
- `audit` is gated at `high` so moderate findings in dev-only transitive dependencies don't block merges, and runs weekly so new advisories surface on their own schedule.

## Dependency overrides

`package.json` declares three `overrides`. All exist to escape advisories that the declaring package pins below the fixed version, with no upstream release available. **These are temporary and should be removed as soon as the condition in the last column is met** — each one forces a version outside a range some dependency declares, so none should outlive its reason.

| Override | Declared range it escapes | Advisory | Remove when |
|---|---|---|---|
| `sharp: ^0.35.3` | `^0.34.5`, by `@huggingface/transformers` | libvips CVEs (GHSA-f88m-g3jw-g9cj) | transformers depends on `sharp >=0.35.0` |
| `adm-zip: ^0.6.0` | `0.5.x`, via `onnxruntime-node@1.24.3` | GHSA-xcpc-8h2w-3j85 | `onnxruntime-node` depends on `adm-zip >=0.6.0` **and** transformers takes that version |
| `esbuild: 0.27.2` | `^0.27.0`, by `tsup` | GHSA-g7r4-m6w7-qqqr (affects `0.27.3` – `0.28.0`) | tsup widens to `^0.28.1` or later |

Two things to know before touching them:

- **`sharp` and `adm-zip` are not covered by `npm test`.** The unit suite mocks providers, so it will stay green even if these break the local ONNX provider. `npm run smoke` is the only guard — it exercises `onnxruntime-node` end-to-end across three models. Re-run it after any change here.
- **`esbuild` must be an override, not just a lockfile entry.** `^0.27.0` has no safe version above `0.27.2`, so anything that re-resolves the range (a fresh `npm install`, `npm update`, a dependency bump) would pull a vulnerable version back in. The pin is what makes it stick.

Check whether any of these can go with:

```bash
npm view @huggingface/transformers dependencies
npm view tsup dependencies.esbuild
```

## TypeScript version constraints

- The project is on **TypeScript 6.0.x**. Do not bump to TypeScript 7 yet: `@typescript-eslint/*` v8 declares `typescript: ">=4.8.4 <6.1.0"`, so `npm run lint` would run against an unsupported compiler.
- `tsconfig.json` sets `"ignoreDeprecations": "6.0"`. This is **required for `npm run build`**, not for our own config — tsup's dts pipeline hardcodes `baseUrl: compilerOptions.baseUrl || "."` (`node_modules/tsup/dist/rollup.js`), and TS 6 raises `TS5101` for deprecated `baseUrl`. Removing the flag breaks the DTS build. It can be dropped once tsup stops injecting `baseUrl`.
