# Code Conventions
> Part of [AGENTS.md](../AGENTS.md) — project guidance for AI coding agents.

## Language & module system

- **Strict TypeScript** (`tsconfig.json`: `strict: true`, target/module `ES2022`/`ESNext`, `moduleResolution: bundler`, `rootDir: src`).
- Source is **ESM-only** (`"type": "module"`). The CJS artifact is produced by the build, not written by hand.
- No `any` at external boundaries — cloud API responses are typed in `src/providers/types.ts`.

## Formatting (Prettier — `.prettierrc`)

- Single quotes, semicolons required, `printWidth: 100`, `trailingComma: "es5"`.
- Run `npm run format` before committing TS changes; `npm run lint` (ESLint with `@typescript-eslint` recommended rules) must pass.

## Public API rules

- A new public function must be exported from its **module barrel** (`src/<module>/index.ts`) and then from the root `src/index.ts`. If it belongs to a grouped namespace (`Math`, `Search`, `Clustering`, `Eval`, `Pipeline`), add it there too.
- New public types/errors go in `src/types.ts` and are re-exported from the root.
- Keep flat named exports stable — they exist for backward compatibility.

## Internal helpers

- Shared, non-public logic lives in `src/internal/` and is annotated `@internal`. Do not re-export internal helpers from the root unless they are genuinely intended as public utilities (only a couple, e.g. `toFloat32`, `isVector`, are).

## Data & dependencies

- Embeddings are `Float32Array` (`Vector`), not `number[]` (v0.3+).
- **Zero production dependencies** — do not add runtime deps. `@huggingface/transformers` is the only (optional) peer dependency and must stay optional; code paths that don't use the local provider must not import it at module top level.

## Git

- Do **not** add Claude Code / AI attribution (`Co-Authored-By`) to commits.
- See `CHANGELOG.md` for versioned history; keep it updated for user-facing changes.
