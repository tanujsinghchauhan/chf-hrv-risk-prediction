from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import serial
from serial import SerialException


@dataclass
class SensorRow:
    time_ms: int
    ir: int
    red: int
    finger: bool
    beat: bool
    bpm: float
    rr_ms: float
    source: str


class ESP32Serial:
    """Read the 8-column CSV format emitted by the ESP32 firmware.

    Expected header:
    time_ms,IR,RED,finger,beat,BPM,RR_ms,source

    Only rows explicitly tagged REAL should be used for HRV features.
    DEMO_SIMULATED rows are display-only.
    """
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection: Optional[serial.Serial] = None
        self.last_error: Optional[str] = None

    def connect(self) -> bool:
        if self.connection and self.connection.is_open:
            return True
        try:
            self.connection = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(1.5)
            self.connection.reset_input_buffer()
            self.last_error = None
            return True
        except (SerialException, OSError) as exc:
            self.connection = None
            self.last_error = str(exc)
            return False

    def close(self) -> None:
        try:
            if self.connection and self.connection.is_open:
                self.connection.close()
        finally:
            self.connection = None

    def read_available(self, max_lines: int = 100) -> list[SensorRow]:
        rows: list[SensorRow] = []
        if not self.connection or not self.connection.is_open:
            return rows
        for _ in range(max_lines):
            try:
                if self.connection.in_waiting <= 0:
                    break
                line = self.connection.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith(("time_ms", "ESP32", "Sensor", "IR,")):
                    continue
                if not line[0].isdigit():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 8:
                    continue
                rows.append(SensorRow(
                    time_ms=int(float(parts[0])),
                    ir=int(float(parts[1])),
                    red=int(float(parts[2])),
                    finger=bool(int(float(parts[3]))),
                    beat=bool(int(float(parts[4]))),
                    bpm=float(parts[5]),
                    rr_ms=float(parts[6]),
                    source=parts[7].upper(),
                ))
            except (ValueError, IndexError, SerialException, OSError) as exc:
                self.last_error = str(exc)
        return rows
