#!/usr/bin/env python3
"""Load test for the /query endpoint.

Measures p50/p95/p99 latency and error rate under concurrency and exits
non-zero if the p95 latency or error rate exceeds the given budgets.

Usage:
    python loadtest/run_loadtest.py --url http://localhost:8000 --requests 500 \
        --concurrency 25 --p95-budget-ms 2000 --max-error-rate 0.01

Auth: set API_KEYS/API_KEY (or --api-key) if the server requires it.
"""

import argparse
import asyncio
import os
import sys
import time

import httpx

PROMPTS = [
    "What is the return policy?",
    "Is Air Max available in size 10?",
    "Check order 1001 status",
    "What are your business hours?",
    "Do you offer international shipping?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Load test /query")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--path", default="/query")
    parser.add_argument("--tenant", default="general")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--api-key", default=os.getenv("API_KEYS") or os.getenv("API_KEY"))
    parser.add_argument("--p95-budget-ms", type=float, default=None)
    parser.add_argument("--max-error-rate", type=float, default=0.05)
    return parser.parse_args()


async def one_request(client, url, payload, headers):
    start = time.perf_counter()
    try:
        response = await client.post(url, json=payload, headers=headers)
        return (time.perf_counter() - start) * 1000, response.status_code
    except httpx.HTTPError:
        return (time.perf_counter() - start) * 1000, None


async def worker(_name, args, results, headers):
    url = args.url.rstrip("/") + args.path
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        while True:
            index = next(counter)
            if index >= args.requests:
                return
            prompt = PROMPTS[index % len(PROMPTS)]
            latency, status = await one_request(
                client,
                url,
                {"message": prompt, "tenant_id": args.tenant, "api_version": "1"},
                headers,
            )
            results.append((latency, status))


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100) * len(ordered) + 0.5)) - 1))
    return ordered[k]


async def run(args):
    global counter
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    results: list[tuple[float, int | None]] = []
    counter = iter(range(args.requests))
    started = time.perf_counter()
    await asyncio.gather(*(worker(i, args, results, headers) for i in range(args.concurrency)))
    wall = time.perf_counter() - started

    latencies = [latency for latency, _ in results]
    errors = [status for _, status in results if status is None or status >= 500]
    print(f"\nrequests: {len(results)}  concurrency: {args.concurrency}  wall: {wall:.2f}s")
    print(f"throughput: {len(results) / wall:.1f} req/s")
    print(f"latency ms  p50: {percentile(latencies, 50):.0f}  "
          f"p95: {percentile(latencies, 95):.0f}  p99: {percentile(latencies, 99):.0f}  "
          f"max: {max(latencies):.0f}")
    print(f"errors: {len(errors)} ({len(errors) / len(results):.1%})")

    failed = False
    if args.p95_budget_ms is not None and percentile(latencies, 95) > args.p95_budget_ms:
        print(f"FAIL: p95 {percentile(latencies, 95):.0f}ms > budget {args.p95_budget_ms}ms")
        failed = True
    if len(errors) / len(results) > args.max_error_rate:
        print(f"FAIL: error rate {len(errors) / len(results):.1%} > {args.max_error_rate:.1%}")
        failed = True
    return 1 if failed else 0


def main():
    args = parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
