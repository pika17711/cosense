import asyncio
import logging
import sys
from mes.CollaborationManager import CollaborationManager
from config import parse_config_file

async def main():
    if len(sys.argv) > 1:
        logging.info("Usage: python main.py [configpath]")
        exit(-1)

    if len(sys.argv) == 1:
        parse_config_file(sys.argv[0])
    
    collaborationManager = CollaborationManager()
    logging.info("collaboration started")
    await collaborationManager.loop()

if __name__ == "__main__":
    asyncio.run(main())