# SPDX-FileCopyrightText: 2019 Carter Nelson for Adafruit Industries
#
# SPDX-License-Identifier: MIT

"""
`neopixel_spi`
================================================================================

SPI driven CircuitPython driver for NeoPixels.


* Author(s): Carter Nelson

Implementation Notes
--------------------

**Hardware:**

* Hardware SPI port required on host platform.

**Software and Dependencies:**

* Adafruit Blinka:
  https://github.com/adafruit/Adafruit_Blinka

* Adafruit's Bus Device library: https://github.com/adafruit/Adafruit_CircuitPython_BusDevice
"""
try:
    from typing import Optional, Tuple, Union

    from busio import SPI
except ImportError:
    pass

import adafruit_pixelbuf
from adafruit_bus_device.spi_device import SPIDevice

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/adafruit/Adafruit_CircuitPython_NeoPixel_SPI.git"

# Pixel color order constants
RGB: str = "RGB"
"""Red Green Blue"""
GRB: str = "GRB"
"""Green Red Blue"""
RGBW: str = "RGBW"
"""Red Green Blue White"""
GRBW: str = "GRBW"
"""Green Red Blue White"""

def duplicate_bits(input_bytes):
    # Create a list to accumulate the bits as integers (0 or 1)
    bit_list = []
    
    # Iterate over each byte in the input
    for byte in input_bytes:
        # Process each bit in the current byte
        for bit_position in range(8):
            bit = (byte >> (7 - bit_position)) & 1
            # Append the duplicated bits to the list
            bit_list.append(bit)
            bit_list.append(bit)
    
    # Create a list to hold the final bytes
    new_bytes_list = []

    # Iterate over the accumulated bit list in chunks of 8
    for i in range(0, len(bit_list), 8):
        # Construct a new byte from each chunk of 8 bits
        new_byte = 0
        for j in range(8):
            if i + j < len(bit_list):
                new_byte |= bit_list[i + j] << (7 - j)
        new_bytes_list.append(new_byte)
    
    # Convert the list of integers back to a bytes object
    return bytes(new_bytes_list)

# Example usage
original = bytes([0b10101010, 0b11001100])
duplicated = duplicate_bits(original)
print("Original:  ", [f'{byte:08b}' for byte in original])
print("Duplicated:", [f'{byte:08b}' for byte in duplicated])
class NeoPixel_SPI(adafruit_pixelbuf.PixelBuf):
    """
    A sequence of neopixels.

    :param ~busio.SPI spi: The SPI bus to output neopixel data on.
    :param int n: The number of neopixels in the chain
    :param int bpp: Bytes per pixel. 3 for RGB and 4 for RGBW pixels.
    :param float brightness: Brightness of the pixels between 0.0 and 1.0 where 1.0 is full
      brightness
    :param bool auto_write: True if the neopixels should immediately change when set. If False,
      ``show`` must be called explicitly.
    :param tuple pixel_order: Set the pixel color channel order. GRBW is set by default.
      pixel_order may be a string or a tuple of integers with values between 0 and 3.
    :param int frequency: SPI bus frequency. For 800kHz NeoPixels, use 6400000 (default).
      For 400kHz, use 3200000.
    :param float reset_time: Reset low level time in seconds. Default is 80e-6.
    :param byte bit0: Bit pattern to set timing for a NeoPixel 0 bit.
    :param byte bit1: Bit pattern to set timing for a NeoPixel 1 bit.

    Example:

    .. code-block:: python

        import board
        import neopixel_spi

        pixels = neopixel_spi.NeoPixel_SPI(board.SPI(), 10)
        pixels.fill(0xff0000)
    """

    def __init__(
        self,
        spi: SPI,
        n: int,
        *,
        bpp: int = 3,
        brightness: float = 1.0,
        auto_write: bool = True,
        pixel_order: Optional[Union[str, Tuple[int, ...]]] = None,
        frequency: int = 20000000,
        reset_time: float = 80e-6,
        bit0: int = 0b11000000,
        bit1: int = 0b11110000
    ) -> None:
        # configure bpp and pixel_order
        if not pixel_order:
            pixel_order = GRB if bpp == 3 else GRBW
        else:
            bpp = len(pixel_order)
            if isinstance(pixel_order, tuple):
                order_list = [RGBW[order] for order in pixel_order]
                pixel_order = "".join(order_list)

        # neopixel stuff
        self._bit0 = bit0
        self._bit1 = bit1
        self._trst = reset_time

        # set up SPI related stuff
        self._spi = SPIDevice(spi, baudrate=frequency)
        with self._spi as spibus:
            try:
                # get actual SPI frequency
                freq = spibus.frequency
            except AttributeError:
                # use nominal
                freq = frequency
        self._reset = bytes([0] * round(freq * self._trst / 8))
        self._spibuf = bytearray(8 * n * bpp)

        # everything else taken care of by base class
        super().__init__(
            size=n, brightness=brightness, byteorder=pixel_order, auto_write=auto_write
        )

    def deinit(self) -> None:
        """Blank out the NeoPixels."""
        self.fill(0)
        self.show()

    def __repr__(self) -> str:
        return "[" + ", ".join([str(x) for x in self]) + "]"

    @property
    def n(self) -> int:
        """
        The number of neopixels in the chain (read-only)
        """
        return len(self)

    def _transmit(self, buffer: bytearray) -> None:
        """Shows the new colors on the pixels themselves if they haven't already
        been autowritten."""
        self._transmogrify(buffer)
        # pylint: disable=no-member
        with self._spi as spi:
            # write out special byte sequence surrounded by RESET
            # leading RESET needed for cases where MOSI rests HI
            #print(self._reset + self._spibuf + self._reset)
            spi.write(self._reset + self._spibuf + self._reset)

    def _transmogrify(self, buffer: bytearray) -> None:
        """Turn every BIT of buf into a special BYTE pattern."""
        k = 0
        
        for i in range(len(buffer)):
            # MSB first
            byte = buffer[i]
            for i in range(7, -1, -1):
                if byte >> i & 0x01:
                    self._spibuf[k] = self._bit1  # A NeoPixel 1 bit
                else:
                    self._spibuf[k] = self._bit0  # A NeoPixel 0 bit
                k += 1