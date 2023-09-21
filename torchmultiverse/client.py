"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

import websockets
import asyncio
import json
import io
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
from PIL import Image
import os
import re
import av
import gc
from functools import reduce
import struct
import functools
import numpy as np
from type_messages import MessageType, ToClientMessageType

def convertIntMessageToByte(messageInt):
    return int(messageInt).to_bytes(1, byteorder='little', signed=False)

def convertByteMessageToInt(messageByte):
    return [int.from_bytes(b, byteorder='little', signed=False) for b in messageByte]

# A generic message has a type and a descriptor.
def emitDescriptor(messageType, descriptor):
    # Convert the dscriptor object into a JSON string.
    descriptorAsString = json.dumps(descriptor, separators=(',', ':'))
    # Add the UTF-16 JSON string to the array byte buffer, going two bytes at
    # a time.
    byte_array = int(messageType).to_bytes(1, byteorder='little', signed=False) + int(len(descriptorAsString)).to_bytes(1, byteorder='little', signed=False) + descriptorAsString.encode("utf-16-be") + b'\x00'
    return byte_array

STRIP_CANDIDATES_RE = re.compile("^a=(candidate:.*|end-of-candidates)\r\n", re.M)

def strip_ice_candidates(description):
    return RTCSessionDescription(
        sdp=STRIP_CANDIDATES_RE.sub("", description.sdp), type=description.type
    )

"""
Websocket Unreal Client definition.
"""
class WebSocketClient():

    def __init__(self, server, outdir, start_index, offset, number_messages, ngpus=1, indexes=None):
        self.connected = False
        self.processing = False
        self.server = server
        self.k = 0
        self.number_messages = number_messages
        self.start_index = start_index
        self.offset = offset
        self.ngpus = ngpus
        self.outdir = outdir
        self.first = True
        self.list_indexes = indexes

    async def connect(self):
        '''
            Connecting to webSocket server

            websockets.client.connect returns a WebSocketClientProtocol, which is used to send and receive messages
        '''
        self.connection = await websockets.connect(self.server)#'ws://100.108.72.243:8000')
        if self.connection.open:
            print('Connection stablished. Client correcly connected')
            return self.connection

    def processFrame(self, frame):
        self.frame += frame[4:]

    def channel_send(self, channel, message):
        # self.channel_log(channel, ">", message)
        channel.send(message)

    def channel_log(self, channel, t, message):
        print("channel(%s) %s %s" % (channel.label, t, message))

    def channel_log_message(self, message):
        """
        Log all messages received by the client from Unreal through the data channel.
        When a FreezeFrame is received by the client, we save the current frame as an image.
        """        
        if message[0] == ToClientMessageType.QualityControlOwnership:
            print("QualityControlOwnership")
        elif message[0] == ToClientMessageType.Command:
            print("Command: ", message[1:].decode("utf-16"))
        # We only listen for FreezeFrame if a complete list of all the factors were sent to unreal
        elif message[0] == ToClientMessageType.FreezeFrame and self.processing:
            # We concatenate the frame that are currently being received by the client
            self.frame = self.frame + message[1:][4:]
            # After receiving all frames, we read them as an Image and save it
            if len(self.frame) == int.from_bytes(message[1:5], byteorder='little', signed = False):
                print("Frame received")
                # Open the stream
                stream = io.BytesIO(self.frame)
                img = Image.open(stream)
                class_folder = self.current_char
                path_full = self.outdir + class_folder
                # Create target directory if it does not exist
                if not os.path.exists(path_full):
                    os.makedirs(path_full)
                # Save the image
                if self.list_indexes is not None:
                    print("Saving " + '{:09d}.png'.format(self.list_indexes[self.k * self.ngpus + self.offset]))
                    img.save(self.outdir + class_folder + "/" +'{:09d}.png'.format(self.list_indexes[self.k * self.ngpus + self.offset]),"PNG")
                else:
                    img.save(self.outdir + class_folder + "/" +'{:09d}.png'.format(self.start_index + self.k * self.ngpus + self.offset),"PNG")
                img.close()
                stream.close()
                del self.frame
                self.frame = None
                del message
                gc.collect()
                # when the frame is saved, we set self.processing back to false to resume the reading of the list of messages.
                self.processing = False
                self.k += 1
                if self.k == self.number_messages:
                     print("End of the list")
                     self.connected = False

        elif message[0] == ToClientMessageType.Response:
            print("Response")
        elif message[0] == ToClientMessageType.UnfreezeFrame:
            print("UnfreezeFrame")
        elif message[0] == ToClientMessageType.LatencyTest:
            print("LatencyTest")
        elif message[0] == ToClientMessageType.InitialSettings:
            print("Init settings")
        elif message[0] == ToClientMessageType.FileExtension:
            print("FileExtension")
        elif message[0] == ToClientMessageType.FileMimeType:
            print("FileMimeType")
        elif message[0] == ToClientMessageType.FileContents:
            print("FileContents")

    async def read_configs(self, loop, list_messages=None):
        """
        Send all messages generated from the config file (list_messages) to Unreal.
        Each element in the list_messages contains a dict with each factors of variation such that
        list_messages[0] = {"World_Name": "GrassLevel", "Character_name": "Elephant: ....}
        So for each message, we pass each of the elements individually to the Unreal server.
        """       
        while True:
            if not self.connected and self.k == self.number_messages:
                break
            # The first command setup the resolution 
            if self.connected and list_messages is not None and not self.processing and self.first:
                print("ready to send messages")
                # Set the target resolution we want to get
                descriptor = { "Resolution": 1, "X": 512, "Y": 512}
                msg = emitDescriptor(MessageType.UIInteraction, descriptor)
                self.channel_send(self.channel, msg)
                # Just increasing the memorizy size
                descriptor = { "PoolSize": 1 }
                msg = emitDescriptor(MessageType.UIInteraction, descriptor)
                self.channel_send(self.channel, msg)
                await asyncio.sleep(2)
                self.first = False
            # Then we go through the entire list of messages to send to Unreal
            if self.connected and list_messages is not None and not self.processing and self.k != self.number_messages:
                # This loop go though each factors in a given message
                for index_m, message in enumerate(list_messages[self.k]):
                    # We change the current character name to be able to save the image in the correct folder
                    if "Character_Name" in message:
                         self.current_char = os.path.basename(message["Character_Name"]).split('.')[0]
                    # Send message
                    msg = emitDescriptor(MessageType.UIInteraction, message)
                    self.channel_send(self.channel, msg)
                # After each of the factors of variations are send, we are reseting the frame variable as well as setting self.processing to true
                # This variable is important to tell the script to not send new messages while the current frame is not saved.
                self.frame = bytearray()
                self.processing = True
            await asyncio.sleep(0.1)

    async def sendMessage(self, message):
        '''
            Sending message to webSocket server
        '''
        await self.connection.send(message)

    async def receiveMessage(self, connection, pc, recorder):
        '''
        Managing the connection with the Unreal Server in addition to 
        receiving all server messages and handling them.
        '''
        nb_remote = 0

        # We don't use aiortc abilities to read the video through WebRTC, so this is actually doing nothing.
        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                print("Receiving %s" % track.kind)
                recorder.addTrack(track)

        # Enabling data channel once the client is connected to the Unreal server.
        @pc.on("datachannel")
        def on_datachannel(channel):
            self.channel_log(channel, "-", "created by remote party")
            self.channel = channel

            @channel.on("message")
            def on_message(message):
                self.channel_log_message(message)
                if isinstance(message, str) and message.startswith("ping"):
                    # reply
                    self.channel_send(channel, "pong" + message[4:])

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            print(f"ICE connection state is {pc.iceConnectionState}")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print("Connection state is %s" % pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()

        """
        Important loop that establish the connection to the Unreal client.
        """
        while True:
            try:
                if pc.connectionState == "connected":
                    print("Client is connected !")
                    self.connected = True
                    await recorder.start()
                # We stop the connection when 
                if not self.connected and self.k == self.number_messages:
                     break
                
                message = await connection.recv()
                message_json = json.loads(message)
                if message_json["type"] == "config":
                    print("Inbound config:", message_json)
                elif message_json["type"] == "playerCount":
                    print("Inbound playerCount:", message_json)
                elif message_json["type"] == "offer":
                    offer = RTCSessionDescription(sdp=message_json["sdp"], type=message_json["type"])
                    await pc.setRemoteDescription(offer)
                elif message_json["type"] == "answer":
                    print("Inbound answer:", message_json)
                    answer = RTCSessionDescription(sdp=message_json["sdp"], type=message_json["type"])
                    await pc.setRemoteDescription(answer)
                elif message_json["type"] == "iceCandidate":
                    candidate = candidate_from_sdp(message_json["candidate"]["candidate"].split(":", 1)[1:][0])
                    candidate.sdpMid = message_json["candidate"]["sdpMid"]
                    candidate.sdpMLineIndex = message_json["candidate"]["sdpMLineIndex"]
                    await pc.addIceCandidate(candidate)
                    nb_remote += 1
                    if nb_remote > 3:
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        new_desc = strip_ice_candidates(pc.localDescription)
                        await self.sendMessage(json.dumps({"type":new_desc.type, "sdp":new_desc.sdp}))
                        await asyncio.sleep(10)
            except websockets.exceptions.ConnectionClosed:
                print('Connection with server closed')
                await recorder.stop()
                break
