"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

from dataclasses import dataclass

# Must be kept in sync with PixelStreamingProtocol::EToUE5Msg C++ enum
@dataclass
class MessageType:
    IFrameRequest: int = 0
    RequestQualityControl: int = 1
    FpsRequest: int = 2
    AverageBitrateRequest: int = 3
    StartStreaming: int = 4
    StopStreaming: int = 5
    LatencyTest: int = 6
    RequestInitialSettings: int = 7
    # Generic Input Messages. Range = 50..59.
    UIInteraction: int = 50
    Command: int = 51
    # Keyboard Input Message. Range = 60..69.
    KeyDown: int = 60
    KeyUp: int = 61
    KeyPress: int = 62
    # Mouse Input Messages. Range = 70..79.
    MouseEnter: int = 70
    MouseLeave: int = 71
    MouseDown: int = 72
    MouseUp: int = 73
    MouseMove: int = 74
    MouseWheel: int = 75
    # Touch Input Messages. Range = 80..89.
    TouchStart: int = 80
    TouchEnd: int = 81
    TouchMove: int = 82
    # Gamepad Input Messages. Range = 90..99
    GamepadButtonPressed: int = 90
    GamepadButtonReleased: int = 91
    GamepadAnalog: int = 92

# Must be kept in sync with PixelStreamingProtocol::EToPlayerMsg C++ enum.
@dataclass
class ToClientMessageType:
    QualityControlOwnership: int = 0
    Response: int = 1
    Command: int = 2
    FreezeFrame: int = 3
    UnfreezeFrame: int = 4
    VideoEncoderAvgQP: int = 5
    LatencyTest: int = 6
    InitialSettings: int = 7
    FileExtension: int = 8
    FileMimeType: int = 9
    FileContents: int = 10
