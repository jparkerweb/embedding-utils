import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { TokenBucketRateLimiter } from '../../src/pipeline/rate-limiter';

describe('TokenBucketRateLimiter', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ── RPM limit ───────────────────────────────────────────────────────────────

  it('allows up to requestsPerMinute requests without delay', async () => {
    const limiter = new TokenBucketRateLimiter({ requestsPerMinute: 3 });

    // All 3 should resolve immediately
    for (let i = 0; i < 3; i++) {
      await limiter.waitForCapacity(0);
    }
  });

  it('delays the request that exceeds RPM limit', async () => {
    const limiter = new TokenBucketRateLimiter({ requestsPerMinute: 2 });

    // Use up both request tokens
    await limiter.waitForCapacity(0);
    await limiter.waitForCapacity(0);

    // Third request should be delayed
    let resolved = false;
    const promise = limiter.waitForCapacity(0).then(() => {
      resolved = true;
    });

    // Should not be resolved yet
    await vi.advanceTimersByTimeAsync(0);
    expect(resolved).toBe(false);

    // Advance 60s to refill
    await vi.advanceTimersByTimeAsync(60_000);
    await promise;
    expect(resolved).toBe(true);
  });

  // ── TPM limit ───────────────────────────────────────────────────────────────

  it('allows requests that fit within tokensPerMinute', async () => {
    const limiter = new TokenBucketRateLimiter({ tokensPerMinute: 100 });

    await limiter.waitForCapacity(50);
    await limiter.waitForCapacity(50);
  });

  it('delays the request that exceeds TPM limit', async () => {
    const limiter = new TokenBucketRateLimiter({ tokensPerMinute: 100 });

    await limiter.waitForCapacity(80);

    let resolved = false;
    const promise = limiter.waitForCapacity(30).then(() => {
      resolved = true;
    });

    await vi.advanceTimersByTimeAsync(0);
    expect(resolved).toBe(false);

    // After 60s, full bucket refills
    await vi.advanceTimersByTimeAsync(60_000);
    await promise;
    expect(resolved).toBe(true);
  });

  // ── Both limits together ────────────────────────────────────────────────────

  it('applies the more restrictive limit when both are set', async () => {
    const limiter = new TokenBucketRateLimiter({
      requestsPerMinute: 10,
      tokensPerMinute: 50,
    });

    // Use up token budget in one call (RPM still has capacity)
    await limiter.waitForCapacity(50);

    let resolved = false;
    const promise = limiter.waitForCapacity(10).then(() => {
      resolved = true;
    });

    await vi.advanceTimersByTimeAsync(0);
    expect(resolved).toBe(false);

    // Tokens refill over time
    await vi.advanceTimersByTimeAsync(60_000);
    await promise;
    expect(resolved).toBe(true);
  });

  // ── Refill ──────────────────────────────────────────────────────────────────

  it('refills tokens proportionally based on elapsed time', async () => {
    const limiter = new TokenBucketRateLimiter({
      requestsPerMinute: 60, // 1 per second refill
      tokensPerMinute: 600, // 10 per second refill
    });

    // Consume all
    for (let i = 0; i < 60; i++) {
      await limiter.waitForCapacity(10);
    }

    // After 1 second: 1 RPM token, 10 TPM tokens refilled
    await vi.advanceTimersByTimeAsync(1_000);
    // Should now be able to make 1 request with 10 tokens
    await limiter.waitForCapacity(10);
  });

  // ── waitForCapacity resolves when capacity is available ─────────────────────

  it('waitForCapacity resolves once capacity becomes available', async () => {
    const limiter = new TokenBucketRateLimiter({ requestsPerMinute: 1 });

    await limiter.waitForCapacity(0);

    const start = Date.now();
    const promise = limiter.waitForCapacity(0).then(() => Date.now());

    // Advance 60s
    await vi.advanceTimersByTimeAsync(60_000);
    const end = await promise;
    expect(end - start).toBeGreaterThanOrEqual(60_000);
  });
});
