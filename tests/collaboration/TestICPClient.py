
import base64
import json
import logging
import time
import zmq


class TestICPClient:
    def __init__(self, client_oid: str):
        # ... 原有初始化代码 ...
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.connect("tcp://127.0.0.1:5556")
        time.sleep(1)
        self.subscribe_to_oid(client_oid)  # 新增订阅方法
        
    def subscribe_to_oid(self, oid: str):
        """向Broker订阅目标oid"""
        subscribe_msg = {
            "type": "subscribe",
            "oid": oid
        }
        self.socket.send_json(subscribe_msg)  # 使用ZMQ原生发送方法

    def recv_message(self):
        """
        接收消息方法：支持带 topic 和不带 topic 的情况，
        并对消息体中的 'coopMap' 字段进行 Base64 解码。
        """
        raw_message_str = ""
        parsed_message_dict = None
        try:
            raw_message_str = self.socket.recv_string()
        except zmq.error.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                logging.debug("ZMQ recv_string would block (EAGAIN). No message.")
                return None
            logging.error(f"ZMQ error receiving string: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during recv_string: {e}")
            return None

        try:

            parsed_message_dict = json.loads(raw_message_str)

            # --- 开始解码 coopMap ---
            if isinstance(parsed_message_dict, dict) and "msg" in parsed_message_dict and \
               isinstance(parsed_message_dict["msg"], dict) and "coopmap" in parsed_message_dict["msg"]:
                
                coopMap_str = parsed_message_dict["msg"]["coopmap"]
                if isinstance(coopMap_str, str) and coopMap_str:
                    try:
                        base64_encoded_bytes = coopMap_str.encode('utf-8')
                        original_coopMap_bytes = base64.b64decode(base64_encoded_bytes)
                        
                        parsed_message_dict["msg"]["coopmap"] = original_coopMap_bytes 
                        logging.info(f"Successfully decoded 'coopMap' field.")
                    except base64.binascii.Error as b64_err:
                        logging.error(f"Failed to Base64 decode 'coopMap' string ('{coopMap_str}'): {b64_err}. Keeping as string.")
                    except UnicodeEncodeError as enc_err:
                        logging.error(f"Failed to encode 'coopMap' string to bytes before Base64 decoding: {enc_err}. Keeping as string.")
                    except Exception as e_decode:
                        logging.error(f"An unexpected error occurred during 'coopMap' decoding: {e_decode}. Keeping as string.")
                elif coopMap_str == "":
                     logging.debug("'coopMap' field is an empty string, no decoding needed.")

            return parsed_message_dict
        
        except json.JSONDecodeError:
            logging.error(f"[!] Failed to decode JSON message: {raw_message_str}")
            return None
        except ValueError: 
            logging.error(f"[!] Malformed message structure (expected topic and JSON or just JSON): {raw_message_str}")
            try:
                parsed_message_dict = json.loads(raw_message_str)
                return parsed_message_dict
            except json.JSONDecodeError:
                logging.error(f"[!] Failed to decode message as plain JSON either: {raw_message_str}")
                return None
            except Exception as e_inner:
                logging.error(f"[!] Unexpected error during fallback JSON parsing: {e_inner}. Original message: {raw_message_str}")
                return None
        except Exception as e_outer:
             logging.error(f"[!] Unexpected error processing received message: {e_outer}. Original message: {raw_message_str}")
             return None