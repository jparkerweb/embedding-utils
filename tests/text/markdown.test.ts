import { describe, it, expect } from 'vitest';
import { chunkByStructure } from '../../src/text/markdown';

// ─────────────────────────────────────────────────────────────────────────────
// Task 3.1: Structure preservation
// ─────────────────────────────────────────────────────────────────────────────

describe('chunkByStructure — structure preservation', () => {
  it('starts a new chunk at heading boundaries', () => {
    const md = `# Heading One

Some text under heading one.

## Heading Two

Some text under heading two.`;
    const chunks = chunkByStructure(md);
    expect(chunks.length).toBeGreaterThanOrEqual(2);
    // First chunk should relate to Heading One
    expect(chunks[0].text).toContain('Heading One');
    // A later chunk should start at Heading Two
    const h2Chunk = chunks.find((c) => c.text.includes('Heading Two'));
    expect(h2Chunk).toBeDefined();
  });

  it('detects all heading levels (# through ######)', () => {
    const md = `# H1
## H2
### H3
#### H4
##### H5
###### H6`;
    const chunks = chunkByStructure(md);
    // Each heading should start a new chunk
    expect(chunks.length).toBe(6);
  });

  it('never splits a fenced code block', () => {
    const code = 'x\n'.repeat(50); // 50 lines of code
    const md = `# Section

\`\`\`js
${code}\`\`\``;
    const chunks = chunkByStructure(md);
    // The code block should be entirely within one chunk
    const codeChunk = chunks.find((c) => c.text.includes('```js'));
    expect(codeChunk).toBeDefined();
    expect(codeChunk!.text).toContain('```');
    // Ensure no other chunk has partial code
    const codeChunks = chunks.filter((c) => c.text.includes('```'));
    expect(codeChunks.length).toBe(1);
  });

  it('keeps bullet lists together when under size limit', () => {
    const md = `# Section

- item one
- item two
- item three`;
    const chunks = chunkByStructure(md);
    const listChunk = chunks.find((c) => c.text.includes('item one'));
    expect(listChunk).toBeDefined();
    expect(listChunk!.text).toContain('item two');
    expect(listChunk!.text).toContain('item three');
  });

  it('keeps numbered lists together when under size limit', () => {
    const md = `# Section

1. first
2. second
3. third`;
    const chunks = chunkByStructure(md);
    const listChunk = chunks.find((c) => c.text.includes('first'));
    expect(listChunk).toBeDefined();
    expect(listChunk!.text).toContain('second');
    expect(listChunk!.text).toContain('third');
  });

  it('keeps markdown tables together', () => {
    const md = `# Section

| Col A | Col B |
|-------|-------|
| 1     | 2     |
| 3     | 4     |`;
    const chunks = chunkByStructure(md);
    const tableChunk = chunks.find((c) => c.text.includes('Col A'));
    expect(tableChunk).toBeDefined();
    expect(tableChunk!.text).toContain('| 3');
  });

  it('tracks nested headings in metadata', () => {
    const md = `# Parent

## Child

Content under child.`;
    const chunks = chunkByStructure(md);
    const childChunk = chunks.find((c) => c.text.includes('Content under child'));
    expect(childChunk).toBeDefined();
    expect(childChunk!.metadata.headings).toEqual(['Parent', 'Child']);
  });

  it('returns empty array for empty input', () => {
    expect(chunkByStructure('')).toEqual([]);
  });

  it('returns empty array for whitespace-only input', () => {
    expect(chunkByStructure('   \n  \n  ')).toEqual([]);
  });

  it('handles plain text (no markdown) as paragraph chunks', () => {
    const text = 'Just some plain text without any markdown syntax.';
    const chunks = chunkByStructure(text);
    expect(chunks.length).toBe(1);
    expect(chunks[0].text).toBe(text);
    expect(chunks[0].metadata.type).toBe('paragraph');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Task 3.2: Chunk metadata
// ─────────────────────────────────────────────────────────────────────────────

describe('chunkByStructure — metadata', () => {
  it('builds heading hierarchy in metadata.headings', () => {
    const md = `# Intro

Some intro text.

## Setup

Setup instructions.`;
    const chunks = chunkByStructure(md);
    const setupChunk = chunks.find((c) => c.text.includes('Setup instructions'));
    expect(setupChunk).toBeDefined();
    expect(setupChunk!.metadata.headings).toEqual(['Intro', 'Setup']);
  });

  it('tracks byte offset for each chunk', () => {
    const md = `# First

Text.

## Second

More text.`;
    const chunks = chunkByStructure(md);
    // First chunk starts at offset 0
    expect(chunks[0].metadata.offset).toBe(0);
    // Later chunks have increasing offsets
    for (let i = 1; i < chunks.length; i++) {
      expect(chunks[i].metadata.offset).toBeGreaterThan(chunks[i - 1].metadata.offset);
    }
  });

  it('classifies paragraph type correctly', () => {
    const md = 'Just a paragraph of text.';
    const chunks = chunkByStructure(md);
    expect(chunks[0].metadata.type).toBe('paragraph');
  });

  it('classifies code type correctly', () => {
    const md = `# Section

\`\`\`
some code
\`\`\``;
    const chunks = chunkByStructure(md);
    const codeChunk = chunks.find((c) => c.metadata.type === 'code');
    expect(codeChunk).toBeDefined();
  });

  it('classifies list type correctly', () => {
    const md = `# Section

- item a
- item b`;
    const chunks = chunkByStructure(md);
    const listChunk = chunks.find((c) => c.metadata.type === 'list');
    expect(listChunk).toBeDefined();
  });

  it('classifies table type correctly', () => {
    const md = `# Section

| A | B |
|---|---|
| 1 | 2 |`;
    const chunks = chunkByStructure(md);
    const tableChunk = chunks.find((c) => c.metadata.type === 'table');
    expect(tableChunk).toBeDefined();
  });

  it('classifies heading type correctly', () => {
    const md = `# Only Heading`;
    const chunks = chunkByStructure(md);
    expect(chunks[0].metadata.type).toBe('heading');
  });

  it('resets deeper headings when a shallower heading appears', () => {
    const md = `# Top

## Sub

### Deep

Content deep.

## Another Sub

Content another.`;
    const chunks = chunkByStructure(md);
    const anotherChunk = chunks.find((c) => c.text.includes('Content another'));
    expect(anotherChunk).toBeDefined();
    // Should have Top > Another Sub, NOT Top > Sub > Deep > Another Sub
    expect(anotherChunk!.metadata.headings).toEqual(['Top', 'Another Sub']);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Task 3.3: maxTokens + overlap with markdown structure
// ─────────────────────────────────────────────────────────────────────────────

describe('chunkByStructure — maxTokens and overlap', () => {
  it('respects maxTokens limit (no chunk exceeds it except code blocks)', () => {
    const paragraphs = Array.from({ length: 10 }, (_, i) =>
      `Paragraph ${i} has several words that add up to a meaningful amount of content here.`,
    ).join('\n\n');
    const md = `# Section\n\n${paragraphs}`;
    const chunks = chunkByStructure(md, { maxTokens: 20 });
    for (const chunk of chunks) {
      if (chunk.metadata.type !== 'code') {
        const wordCount = chunk.text.split(/\s+/).filter((w) => w.length > 0).length;
        const estimatedTokens = Math.ceil(wordCount * 1.3);
        // Allow some tolerance (the limit is approximate)
        expect(estimatedTokens).toBeLessThanOrEqual(30);
      }
    }
  });

  it('produces overlapping text when overlap option is set', () => {
    const sentences = Array.from({ length: 20 }, (_, i) =>
      `Sentence number ${i} contains some words.`,
    ).join(' ');
    const md = `# Section\n\n${sentences}`;
    const chunks = chunkByStructure(md, { maxTokens: 30, overlap: 10 });
    if (chunks.length >= 2) {
      // Find two consecutive non-heading chunks
      for (let i = 0; i < chunks.length - 1; i++) {
        if (chunks[i].metadata.type !== 'heading' && chunks[i + 1].metadata.type !== 'heading') {
          const wordsA = chunks[i].text.split(/\s+/);
          const wordsB = chunks[i + 1].text.split(/\s+/);
          const tailA = wordsA.slice(-5);
          const headB = wordsB.slice(0, 5);
          const shared = tailA.filter((w) => headB.includes(w));
          // With overlap, at least some words should repeat
          expect(shared.length).toBeGreaterThan(0);
          break;
        }
      }
    }
  });

  it('keeps large code block whole even if it exceeds maxTokens', () => {
    const codeLines = Array.from({ length: 30 }, (_, i) => `  line${i}();`).join('\n');
    const md = `# Code

\`\`\`js
${codeLines}
\`\`\``;
    const chunks = chunkByStructure(md, { maxTokens: 10 });
    const codeChunk = chunks.find((c) => c.metadata.type === 'code');
    expect(codeChunk).toBeDefined();
    // All 30 lines should be in that one chunk
    expect(codeChunk!.text).toContain('line0()');
    expect(codeChunk!.text).toContain('line29()');
  });

  it('splits long paragraph at sentence boundaries when exceeding maxTokens', () => {
    const longParagraph =
      'First sentence is here. Second sentence follows. Third sentence continues. ' +
      'Fourth sentence appears. Fifth sentence ends.';
    const md = `# Intro\n\n${longParagraph}`;
    const chunks = chunkByStructure(md, { maxTokens: 10 });
    // Should split into multiple paragraph chunks
    const paraChunks = chunks.filter((c) => c.metadata.type === 'paragraph');
    expect(paraChunks.length).toBeGreaterThan(1);
    // Each paragraph chunk should end at a sentence boundary (period)
    for (const c of paraChunks) {
      const trimmed = c.text.trim();
      if (trimmed.length > 0) {
        expect(trimmed.endsWith('.')).toBe(true);
      }
    }
  });
});
