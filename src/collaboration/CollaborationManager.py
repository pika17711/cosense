import asyncio
import logging
import concurrent.futures
from config import AppConfig
from collaboration.messageHandler import MessageHandler
from perception.perception_client import PerceptionClient
from utils import InfoDTO
from utils.common import sync_to_async

class CollaborationManager:
    def __init__(self):
        self.perception_client = PerceptionClient()
        self.message_handler = MessageHandler(self.perception_client)
        self.broadcastpub_event = asyncio.Event()
        self.broadcastsub_event = asyncio.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.running = True

    def force_close(self):
        self.executor.shutdown()
        self.message_handler.force_close()
        self.running = False

    async def loop(self):
        tasks = [self.message_handler.recv_loop(), 
                 self.broadcastpub_loop(), 
                 self.broadcastsub_loop(), 
                 self.command_loop(),
                 self.subscribed_send_loop()]
        await asyncio.gather(*tasks)

    async def handle_command(self, argv):
        logging.debug(f"输入的命令是: {argv}")
        if len(argv) == 0:
            pass
        elif len(argv) == 1 and argv[0] == 'exit':
            return False
        elif len(argv) == 2 and argv[0] == 'bpub':
            if argv[1] == 'open':
                self.broadcastpub_open()
                print('ok')
            elif argv[1] == 'close':
                self.broadcastpub_close()
                print('ok')
            else:
                print('syntax error')
        elif len(argv) == 2 and argv[0] == 'bsub':
            if argv[1] == 'open':
                self.broadcastsub_open()
                print('ok')
            elif argv[1] == 'close':
                self.broadcastsub_close()
                print('ok')
            else:
                print('syntax error')
        elif len(argv) == 2 and argv[0] == 'show':
            if argv[1] == 'subing':
                print([cctx.remote_id() for cctx in self.message_handler.get_subscribing()])
            elif argv[1] == 'subed':
                print([cctx.remote_id() for cctx in self.message_handler.get_subscribed()])
        else:
            print('syntax error')
        return True

    async def command_loop(self):
        while self.running:
            try:
                # 异步读取用户输入
                print("$ ", end='')
                command = await asyncio.get_event_loop().run_in_executor(self.executor, input)
                argv = command.split()
                should_continue = await self.handle_command(argv)
                if not should_continue:
                    self.force_close()
                    break
            except EOFError:
                break

    def broadcastpub_open(self):
        self.broadcastpub_event.set()

    def broadcastpub_close(self):
        self.broadcastpub_event.clear()

    async def broadcastpub_send(self):
        await self.message_handler.broadcastpub_send()

    async def broadcastpub_loop(self):
        while self.running:
            if self.broadcastpub_event.is_set():
                await self.message_handler.broadcastpub_send()
                await asyncio.sleep(AppConfig.broadcastpub_period/1000)
            else:
                await self.broadcastpub_event.wait()

    def broadcastsub_open(self):
        self.broadcastsub_event.set()

    def broadcastsub_close(self):
        self.broadcastsub_event.clear()

    async def broadcastsub_send(self):
        await self.message_handler.broadcastsub_send()

    async def broadcastsub_loop(self):
        while self.running:
            if self.broadcastsub_event.is_set():
                await self.message_handler.broadcastsub_send()
                await asyncio.sleep(AppConfig.broadcastsub_period/1000)
            else:
                await self.broadcastsub_event.wait()

    def get_all_data(self):
        ts1, pose, velocity, acceleration = self.perception_client.get_my_pva_info()
        ts2, extrinsic_matrix = self.perception_client.get_my_extrinsic_matrix()
        ts3, feat = self.perception_client.get_my_feature()
        infodto = InfoDTO.InfoDTO(1, AppConfig.id, extrinsic_matrix, None, None, feat, ts3, velocity, ts1, pose, ts1, acceleration, ts2, None, None)
        data = InfoDTO.InfoDTOSerializer.serialize(infodto)
        return data

    async def subscribed_send_loop(self):
        logging.info("订阅者数据发送循环启动")
        while self.running:
            subeds = self.message_handler.get_subscribed()
            data = await sync_to_async(self.get_all_data)()
            logging.info(f"订阅者数据发送, 订阅者列表{list(subeds)}, 发送数据 {len(data)}B")
            for cctx in subeds:
                asyncio.create_task(cctx.send_data(data))
            await asyncio.sleep(AppConfig.send_data_period/1000)