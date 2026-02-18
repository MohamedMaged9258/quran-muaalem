import asyncio
import aiohttp
from pathlib import Path


async def test():
    audio_path = Path("assets/test.wav")
    audio_bytes = audio_path.read_bytes()

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/predict",
            data={"audio": audio_bytes},
        ) as resp:
            result = await resp.json()
            print(result)


if __name__ == "__main__":
    asyncio.run(test())
