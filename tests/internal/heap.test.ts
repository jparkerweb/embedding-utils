import { describe, it, expect } from 'vitest';
import { MinHeap } from '../../src/internal/heap';

describe('MinHeap', () => {
  it('pops elements in ascending order (min first)', () => {
    const heap = new MinHeap<number>((a, b) => a - b);
    [5, 3, 8, 1, 4].forEach((n) => heap.push(n));
    const result: number[] = [];
    while (heap.size > 0) result.push(heap.pop()!);
    expect(result).toEqual([1, 3, 4, 5, 8]);
  });

  it('returns undefined on empty pop', () => {
    const heap = new MinHeap<number>((a, b) => a - b);
    expect(heap.pop()).toBeUndefined();
  });

  it('returns undefined on empty peek', () => {
    const heap = new MinHeap<number>((a, b) => a - b);
    expect(heap.peek()).toBeUndefined();
  });

  it('peek returns the minimum without removing', () => {
    const heap = new MinHeap<number>((a, b) => a - b);
    heap.push(10);
    heap.push(3);
    expect(heap.peek()).toBe(3);
    expect(heap.size).toBe(2);
  });

  it('tracks size correctly', () => {
    const heap = new MinHeap<number>((a, b) => a - b);
    expect(heap.size).toBe(0);
    heap.push(1);
    expect(heap.size).toBe(1);
    heap.push(2);
    expect(heap.size).toBe(2);
    heap.pop();
    expect(heap.size).toBe(1);
  });

  it('maintains min-heap property with 1000 random inserts', () => {
    const heap = new MinHeap<number>((a, b) => a - b);
    const values: number[] = [];
    for (let i = 0; i < 1000; i++) {
      const v = Math.random() * 10000;
      values.push(v);
      heap.push(v);
    }
    values.sort((a, b) => a - b);
    const result: number[] = [];
    while (heap.size > 0) result.push(heap.pop()!);
    expect(result.length).toBe(1000);
    for (let i = 1; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(result[i - 1]);
    }
  });

  it('works with custom comparator (max-heap via inverted comparator)', () => {
    const heap = new MinHeap<number>((a, b) => b - a);
    [5, 3, 8, 1, 4].forEach((n) => heap.push(n));
    expect(heap.pop()).toBe(8);
    expect(heap.pop()).toBe(5);
  });

  it('handles single element', () => {
    const heap = new MinHeap<number>((a, b) => a - b);
    heap.push(42);
    expect(heap.peek()).toBe(42);
    expect(heap.pop()).toBe(42);
    expect(heap.size).toBe(0);
  });
});
