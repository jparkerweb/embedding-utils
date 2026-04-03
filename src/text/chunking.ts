/**
 * Splits text into chunks of approximately `maxTokens` tokens.
 *
 * Token estimation uses a ~1.3 tokens per word heuristic (not tokenizer-accurate).
 * Splits on word boundaries only — never mid-word.
 *
 * @param text - Input text to split
 * @param maxTokens - Maximum estimated tokens per chunk
 * @param options - Optional overlap and separator settings
 * @returns Array of text chunks
 */
export function chunkByTokenCount(
  text: string,
  maxTokens: number,
  options?: { overlap?: number; separator?: string },
): string[] {
  if (!text) return [];

  const separator = options?.separator ?? ' ';
  const overlap = options?.overlap ?? 0;
  const words = text.split(/\s+/).filter((w) => w.length > 0);
  if (words.length === 0) return [];

  const tokensPerWord = 1.3;
  const maxWords = Math.max(1, Math.floor(maxTokens / tokensPerWord));
  const overlapWords = Math.max(0, Math.floor(overlap / tokensPerWord));

  const chunks: string[] = [];
  let start = 0;

  while (start < words.length) {
    const end = Math.min(start + maxWords, words.length);
    chunks.push(words.slice(start, end).join(separator));
    const step = maxWords - overlapWords;
    start += step > 0 ? step : maxWords;
  }

  return chunks;
}

/**
 * Splits text on sentence boundaries using `Intl.Segmenter`, then groups
 * sentences into chunks that fit within `maxTokens` (estimated at ~1.3 tokens/word).
 *
 * If `maxTokens` is not provided, each sentence is returned as its own chunk.
 *
 * @param text - Input text to split
 * @param options - Optional maxTokens and overlap settings
 * @returns Array of text chunks (each containing one or more sentences)
 * @throws If `Intl.Segmenter` is not available in the runtime
 */
export function chunkBySentence(
  text: string,
  options?: { maxTokens?: number; overlap?: number },
): string[] {
  if (!text) return [];

  if (typeof Intl === 'undefined' || !('Segmenter' in Intl)) {
    throw new Error(
      'chunkBySentence requires Intl.Segmenter (Node 16+ or modern browser)',
    );
  }

  const segmenter = new Intl.Segmenter(undefined, { granularity: 'sentence' });
  const sentences = Array.from(segmenter.segment(text), (s) => s.segment).filter(
    (s) => s.trim().length > 0,
  );

  if (sentences.length === 0) return [];

  const maxTokens = options?.maxTokens;
  if (maxTokens === undefined) return sentences;

  const overlap = options?.overlap ?? 0;
  const tokensPerWord = 1.3;

  function estimateTokens(s: string): number {
    const wordCount = s.split(/\s+/).filter((w) => w.length > 0).length;
    return Math.ceil(wordCount * tokensPerWord);
  }

  const chunks: string[] = [];
  let currentChunk: string[] = [];
  let currentTokens = 0;

  for (const sentence of sentences) {
    const sentenceTokens = estimateTokens(sentence);

    if (currentChunk.length > 0 && currentTokens + sentenceTokens > maxTokens) {
      chunks.push(currentChunk.join(''));

      // Handle overlap: keep trailing sentences that fit within overlap tokens
      if (overlap > 0) {
        let overlapTokens = 0;
        let overlapStart = currentChunk.length;
        for (let i = currentChunk.length - 1; i >= 0; i--) {
          const t = estimateTokens(currentChunk[i]);
          if (overlapTokens + t > overlap) break;
          overlapTokens += t;
          overlapStart = i;
        }
        currentChunk = currentChunk.slice(overlapStart);
        currentTokens = overlapTokens;
      } else {
        currentChunk = [];
        currentTokens = 0;
      }
    }

    currentChunk.push(sentence);
    currentTokens += sentenceTokens;
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join(''));
  }

  return chunks;
}
