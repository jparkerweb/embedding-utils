/**
 * Seeded PRNG using xorshift32.
 * @internal
 */
export function createRng(seed: number): () => number {
  let state = seed | 0 || 1; // ensure non-zero
  return () => {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

/**
 * Returns a new array with elements shuffled using Fisher-Yates with seeded RNG.
 * @internal
 */
export function shuffleArray<T>(arr: T[], seed: number): T[] {
  const result = arr.slice();
  const rng = createRng(seed);
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
