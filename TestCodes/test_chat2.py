import httpx, json, asyncio

async def test():
    body = {"question": "explain electric charge", "class_number": "12", "subject": "Physics", "history": []}
    async with httpx.AsyncClient(timeout=30) as c:
        async with c.stream("POST", "http://localhost:8000/api/chat", json=body) as resp:
            print("status:", resp.status_code)
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "images" in data:
                        print(f"IMAGES ({len(data['images'])}):")
                        for img in data["images"]:
                            print(f"  page={img['page_number']} path={img['image_path']}")
                    elif "error" in data:
                        print("ERROR:", data["error"])
                    elif "done" in data:
                        print("DONE")
                        break

asyncio.run(test())
