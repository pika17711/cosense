import asyncio
from mes.CollaborationManager import CollaborationManager


if __name__ == "__main__":

    collaborationManager = CollaborationManager()
    asyncio.run(collaborationManager.loop())