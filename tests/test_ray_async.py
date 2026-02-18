import asyncio
import aiohttp
import time
import sys
from pathlib import Path


async def send_request(session, url, audio_bytes, request_id):
    """Send one request and return its latency in seconds, or None on failure."""
    start = time.perf_counter()
    try:
        async with session.post(url, data={"audio": audio_bytes}) as resp:
            # Wait for the response and optionally print the result
            result = await resp.json()
            # Uncomment next line to see each response
            # print(f"Request {request_id}: {result}")
            end = time.perf_counter()
            return end - start
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return None


async def main():
    # Determine number of requests (N)
    N = 10
    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
        except ValueError:
            print("Usage: python script.py [N]")
            sys.exit(1)

    url = "http://localhost:8000/predict"
    audio_path = Path("assets/test.wav")

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)

    audio_bytes = audio_path.read_bytes()

    async with aiohttp.ClientSession() as session:
        print(f"Sending {N} concurrent requests to {url} ...")
        tasks = [send_request(session, url, audio_bytes, i) for i in range(N)]

        overall_start = time.perf_counter()
        latencies = await asyncio.gather(*tasks)
        overall_end = time.perf_counter()

    # Filter out failed requests (None values)
    successful_latencies = [l for l in latencies if l is not None]
    successful_count = len(successful_latencies)

    if successful_count == 0:
        print("No successful requests.")
        return

    total_time = overall_end - overall_start
    throughput = successful_count / total_time

    avg_latency = sum(successful_latencies) / successful_count
    min_latency = min(successful_latencies)
    max_latency = max(successful_latencies)

    print("\n--- Results ---")
    print(f"Successful requests: {successful_count}/{N}")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Throughput: {throughput:.2f} requests/second")
    print(f"Average latency: {avg_latency * 1000:.2f} ms")
    print(f"Min latency: {min_latency * 1000:.2f} ms")
    print(f"Max latency: {max_latency * 1000:.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())

