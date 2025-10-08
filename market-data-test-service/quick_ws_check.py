import asyncio, json, os, websockets

async def main():
    uri = os.environ["WS_URL"]
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"action":"subscribe","symbols":["AAPL"]}))
        print("ACK:", await asyncio.wait_for(ws.recv(), timeout=2))
        print("NEXT:", await asyncio.wait_for(ws.recv(), timeout=5))  # should include symbol & price

asyncio.run(main())
