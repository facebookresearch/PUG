
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

import websockets
import asyncio
from aiortc import (
    RTCIceCandidate,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaBlackhole
from functools import reduce
from dataclasses import dataclass
import argparse
from client import WebSocketClient
from json_parser import process_json

if __name__ == '__main__':

    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-s", "--server", help = "Url to the server", required=True)
    parser.add_argument("-o", "--outdir", help = "Where to save the images", required=True)
    parser.add_argument("--index", help = "Which config to use", type=str, default='')
    parser.add_argument("--offset", help = "Which gpu number", type=int, default=0)
    parser.add_argument("--ngpus", help = "How many gpus to use", type=int, default=1)
    # Read arguments from command line
    args = parser.parse_args()

    # Paths to the config to use to samples the data
    liste_configs = ["config_animal.json"] 
    offset = 0
    start_index = 0
    list_messages = []
    for idx_config, config in enumerate(liste_configs):
      # Get config
      list_messages = process_json(config, list_start_indexes[idx_config], list_folders[idx_config])

      print("Number of images to sample: ", len(list_messages))

      # Uncomment the following to sample across different gpu. In that instance it will samples 1000 images by gpu.
      """
      list_messages_per_gpu = list_messages[1000*args.offset:1000*args.offset+1000]
      start_index = 1000*args.offset
      print("Total", len(list_messages_per_gpu))
      """

      # Creating client object
      client = WebSocketClient(args.server, args.outdir, start_index, 0, len(list_messages_per_gpu), args.ngpus)
      loop = asyncio.get_event_loop()
      recorder = MediaBlackhole()

      pc = RTCPeerConnection()
      # Start connection and get client connection protocol
      connection = loop.run_until_complete(client.connect())
      # Start listener and heartbeat 
      tasks = [
        asyncio.ensure_future(client.receiveMessage(connection, pc, recorder)),
        asyncio.ensure_future(client.read_configs(loop, list_messages_per_gpu)),
      ]

      try:
        loop.run_until_complete(asyncio.wait(tasks))
      except KeyboardInterrupt:
        pass
      finally:
        # cleanup
        loop.run_until_complete(pc.close())

      start_index = start_index + len(list_messages)
