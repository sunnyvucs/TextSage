import httpx, json, asyncio

async def test():
    body = {"question": "What is photosynthesis?", "class_number": "12", "subject": "Biology", "history": []}
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post("http://localhost:8000/api/chat", json=body)
        print("status:", r.status_code)
        print("body:", r.text[:2000])

asyncio.run(test())
