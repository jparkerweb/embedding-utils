/**
 * Token bucket rate limiter for controlling request and token throughput.
 *
 * Uses dual buckets (RPM and TPM) to enforce rate limits. Callers
 * await `waitForCapacity()` before making API calls.
 */
export class TokenBucketRateLimiter {
  private readonly maxRequests: number;
  private readonly maxTokens: number;
  private requestTokens: number;
  private tokenBucket: number;
  private lastRefill: number;

  constructor(options: { requestsPerMinute?: number; tokensPerMinute?: number } = {}) {
    this.maxRequests = options.requestsPerMinute ?? Infinity;
    this.maxTokens = options.tokensPerMinute ?? Infinity;
    this.requestTokens = this.maxRequests;
    this.tokenBucket = this.maxTokens;
    this.lastRefill = performance.now();
  }

  /**
   * Wait until both RPM and TPM buckets have sufficient capacity,
   * then consume the tokens.
   */
  async waitForCapacity(requestTokenCount: number): Promise<void> {
    for (;;) {
      this.refill();

      const hasRequestCapacity = this.requestTokens >= 1;
      const hasTokenCapacity = this.tokenBucket >= requestTokenCount;

      if (hasRequestCapacity && hasTokenCapacity) {
        this.requestTokens -= 1;
        this.tokenBucket -= requestTokenCount;
        return;
      }

      // Calculate wait time until enough capacity refills
      const waitMs = this.calculateWaitMs(requestTokenCount);
      await this.sleep(waitMs);
    }
  }

  private refill(): void {
    const now = performance.now();
    const elapsed = now - this.lastRefill;
    if (elapsed <= 0) return;

    // Refill proportionally: capacity per minute * (elapsed / 60000)
    const fraction = elapsed / 60_000;

    this.requestTokens = Math.min(
      this.maxRequests,
      this.requestTokens + this.maxRequests * fraction,
    );
    this.tokenBucket = Math.min(
      this.maxTokens,
      this.tokenBucket + this.maxTokens * fraction,
    );

    this.lastRefill = now;
  }

  private calculateWaitMs(requestTokenCount: number): number {
    let wait = 0;

    if (this.requestTokens < 1 && this.maxRequests !== Infinity) {
      const deficit = 1 - this.requestTokens;
      wait = Math.max(wait, (deficit / this.maxRequests) * 60_000);
    }

    if (this.tokenBucket < requestTokenCount && this.maxTokens !== Infinity) {
      const deficit = requestTokenCount - this.tokenBucket;
      wait = Math.max(wait, (deficit / this.maxTokens) * 60_000);
    }

    // Minimum 10ms to avoid busy-spinning
    return Math.max(wait, 10);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
