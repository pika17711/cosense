import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import asyncio
import logging
from collaboration.CollaborationManager import CollaborationManager
from config import parse_config_file

async def main():
    if len(sys.argv) > 1:
        logging.info("Usage: python main.py [configpath]")
        exit(-1)

    if len(sys.argv) == 1:
        parse_config_file(sys.argv[0])

    collaborationManager = CollaborationManager()
    logging.basicConfig(level=logging.DEBUG, 
                        filename='collaboration.log',
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("协同模块启动")
    
    try:
        await collaborationManager.loop()
    except KeyboardInterrupt:
        collaborationManager.force_close()
        logging.info("接收到 Ctrl + C，程序退出。")

if __name__ == "__main__":
    asyncio.get_event_loop().set_debug(True)
    asyncio.run(main(), debug=True)