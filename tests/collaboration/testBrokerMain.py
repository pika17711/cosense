
import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from tests.collaboration.testBroker import TestBroker

if __name__ == "__main__":
    test_broker = TestBroker()
    test_broker.run()