import type { StructuredChunk, StructuredChunkType } from '../types';

// ─────────────────────────────────────────────────────────────────────────────
// Internal block parser
// ─────────────────────────────────────────────────────────────────────────────

interface Block {
  type: StructuredChunkType;
  text: string;
  /** Heading level (1-6) when type === 'heading', undefined otherwise. */
  headingLevel?: number;
  /** Heading text (without `#` prefix) when type === 'heading'. */
  headingText?: string;
  /** Character offset in the original input. */
  offset: number;
}

const HEADING_RE = /^(#{1,6})\s+(.+)$/;
const FENCE_RE = /^(`{3,}|~{3,})/;
const LIST_ITEM_RE = /^\s*(?:[-*+]|\d+\.)\s/;
const TABLE_ROW_RE = /^\|/;

/**
 * Splits markdown text into structural blocks (headings, code fences,
 * lists, tables, paragraphs). Does not attempt full AST parsing — uses
 * line-level regex detection for common GFM patterns.
 */
function parseBlocks(text: string): Block[] {
  const lines = text.split('\n');
  const blocks: Block[] = [];
  let i = 0;

  /** Current character offset tracking. */
  let charOffset = 0;

  function advanceOffset(lineIndex: number): number {
    // Offset is the sum of all previous line lengths + newline chars
    let offset = 0;
    for (let j = 0; j < lineIndex; j++) {
      offset += lines[j].length + 1; // +1 for newline
    }
    return offset;
  }

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // Skip blank lines
    if (trimmed === '') {
      i++;
      continue;
    }

    charOffset = advanceOffset(i);

    // Heading
    const headingMatch = trimmed.match(HEADING_RE);
    if (headingMatch) {
      blocks.push({
        type: 'heading',
        text: line,
        headingLevel: headingMatch[1].length,
        headingText: headingMatch[2].trim(),
        offset: charOffset,
      });
      i++;
      continue;
    }

    // Fenced code block
    const fenceMatch = trimmed.match(FENCE_RE);
    if (fenceMatch) {
      const fenceChar = fenceMatch[1][0]; // ` or ~
      const fenceLen = fenceMatch[1].length;
      const codeLines: string[] = [line];
      i++;
      while (i < lines.length) {
        codeLines.push(lines[i]);
        const closingMatch = lines[i].trim().match(new RegExp(`^${fenceChar}{${fenceLen},}$`));
        if (closingMatch) {
          i++;
          break;
        }
        i++;
      }
      blocks.push({
        type: 'code',
        text: codeLines.join('\n'),
        offset: charOffset,
      });
      continue;
    }

    // Table rows
    if (TABLE_ROW_RE.test(trimmed)) {
      const tableLines: string[] = [];
      while (i < lines.length && TABLE_ROW_RE.test(lines[i].trim())) {
        tableLines.push(lines[i]);
        i++;
      }
      blocks.push({
        type: 'table',
        text: tableLines.join('\n'),
        offset: charOffset,
      });
      continue;
    }

    // List items
    if (LIST_ITEM_RE.test(line)) {
      const listLines: string[] = [];
      while (i < lines.length) {
        const l = lines[i];
        // Continue list: list item, continuation (indented), or blank within list
        if (LIST_ITEM_RE.test(l)) {
          listLines.push(l);
          i++;
        } else if (l.trim() === '' && i + 1 < lines.length && LIST_ITEM_RE.test(lines[i + 1])) {
          // Blank line between list items
          listLines.push(l);
          i++;
        } else {
          break;
        }
      }
      blocks.push({
        type: 'list',
        text: listLines.join('\n'),
        offset: charOffset,
      });
      continue;
    }

    // Paragraph: collect consecutive non-blank, non-special lines
    const paraLines: string[] = [];
    while (i < lines.length) {
      const l = lines[i];
      const t = l.trim();
      if (
        t === '' ||
        HEADING_RE.test(t) ||
        FENCE_RE.test(t) ||
        TABLE_ROW_RE.test(t) ||
        LIST_ITEM_RE.test(l)
      ) {
        break;
      }
      paraLines.push(l);
      i++;
    }
    if (paraLines.length > 0) {
      blocks.push({
        type: 'paragraph',
        text: paraLines.join('\n'),
        offset: charOffset,
      });
    }
  }

  return blocks;
}

// ─────────────────────────────────────────────────────────────────────────────
// Token estimation
// ─────────────────────────────────────────────────────────────────────────────

const TOKENS_PER_CHAR = 0.25; // ~4 chars per token

function estimateTokens(text: string): number {
  return Math.ceil(text.length * TOKENS_PER_CHAR);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sentence splitting for oversized paragraphs
// ─────────────────────────────────────────────────────────────────────────────

function splitSentences(text: string): string[] {
  // Split on ". " or ".\n" while keeping the period with the sentence
  const parts = text.split(/(?<=\.)\s+/);
  return parts.filter((s) => s.trim().length > 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Splits markdown text into structural chunks that respect document structure
 * (headings, code blocks, lists, tables) and attaches heading metadata per chunk.
 *
 * @param text - Markdown text to chunk
 * @param options - Optional maxTokens limit and overlap
 * @returns Array of structured chunks with metadata
 */
export function chunkByStructure(
  text: string,
  options?: { maxTokens?: number; overlap?: number },
): StructuredChunk[] {
  if (!text || text.trim().length === 0) return [];

  const maxTokens = options?.maxTokens;
  const overlap = options?.overlap ?? 0;
  const blocks = parseBlocks(text);

  if (blocks.length === 0) return [];

  const chunks: StructuredChunk[] = [];
  /** Current heading hierarchy — indexed by level (1-6). */
  const headingStack: (string | undefined)[] = new Array(7).fill(undefined);

  for (const block of blocks) {
    // Update heading stack
    if (block.type === 'heading' && block.headingLevel !== undefined && block.headingText) {
      const level = block.headingLevel;
      headingStack[level] = block.headingText;
      // Clear deeper headings
      for (let l = level + 1; l <= 6; l++) {
        headingStack[l] = undefined;
      }
    }

    // Build current headings array from stack
    const headings: string[] = [];
    for (let l = 1; l <= 6; l++) {
      if (headingStack[l] !== undefined) {
        headings.push(headingStack[l]!);
      }
    }

    // Code blocks, lists, tables are never split — emit as-is
    if (block.type === 'code' || block.type === 'list' || block.type === 'table') {
      chunks.push({
        text: block.text,
        metadata: {
          headings: [...headings],
          offset: block.offset,
          type: block.type,
        },
      });
      continue;
    }

    // Headings are emitted if they stand alone (no content follows merged in)
    if (block.type === 'heading') {
      chunks.push({
        text: block.text,
        metadata: {
          headings: [...headings],
          offset: block.offset,
          type: 'heading',
        },
      });
      continue;
    }

    // Paragraph: may need splitting if maxTokens is set
    if (maxTokens && estimateTokens(block.text) > maxTokens) {
      const sentences = splitSentences(block.text);
      let currentText = '';
      let currentOffset = block.offset;

      for (const sentence of sentences) {
        if (currentText.length > 0 && estimateTokens(currentText + ' ' + sentence) > maxTokens) {
          // Emit current accumulated text
          chunks.push({
            text: currentText,
            metadata: {
              headings: [...headings],
              offset: currentOffset,
              type: 'paragraph',
            },
          });

          // Handle overlap
          if (overlap > 0) {
            const overlapSentences = splitSentences(currentText);
            let overlapText = '';
            for (let j = overlapSentences.length - 1; j >= 0; j--) {
              const candidate = overlapSentences[j] + (overlapText ? ' ' + overlapText : '');
              if (estimateTokens(candidate) > overlap) break;
              overlapText = candidate;
            }
            currentOffset = block.offset + text.indexOf(currentText, currentOffset - block.offset) + currentText.length - overlapText.length;
            currentText = overlapText;
          } else {
            currentOffset = block.offset + block.text.indexOf(sentence, currentText.length);
            currentText = '';
          }
        }

        currentText = currentText ? currentText + ' ' + sentence : sentence;
      }

      if (currentText.trim().length > 0) {
        chunks.push({
          text: currentText,
          metadata: {
            headings: [...headings],
            offset: currentOffset,
            type: 'paragraph',
          },
        });
      }
    } else {
      chunks.push({
        text: block.text,
        metadata: {
          headings: [...headings],
          offset: block.offset,
          type: 'paragraph',
        },
      });
    }
  }

  return chunks;
}
