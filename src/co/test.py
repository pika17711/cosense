import server
import sys
import os
def test():
    os.system('ros_debug_publisher.py --topic_name test_pc_publisher1')
    os.system('server.py')


if __name__ == '__main__':
    test()