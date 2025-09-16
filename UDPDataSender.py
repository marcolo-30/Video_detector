import socket
import struct
import math

import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure ThreadPoolExecutor to use a maximum of 12 threads
executor = ThreadPoolExecutor(max_workers=6)  # Adjust max_workers as needed

# Constants
HEADER_SIZE = 20  # Size of the header: total_packets (4 bytes), packet_index (4 bytes), frame_id (4 bytes)
MAX_PAYLOAD_SIZE = 1350 - HEADER_SIZE  # Max UDP payload size minus header size
PORT = 5005


class PacketHeader:
    # Header structure: total_packets (uint32), packet_index (uint32), frame_index (uint32)
    def __init__(self, total_packets, packet_index, frame_index, object_index,total_objects):
        self.total_packets = total_packets
        self.packet_index = packet_index
        self.frame_index = frame_index
        self.object_index = object_index
        self.total_objects = total_objects

    def pack(self):
        # Pack the header as a binary struct
        return struct.pack('<IIIII', self.total_packets, self.packet_index,
                           self.frame_index, self.object_index, self.total_objects)

async def send_data_with_headers_async(host, port, buffer, frame_id, object_id, total_objects):
    """
    Asynchronous wrapper for send_data_with_headers to run in a separate thread.
    :param host: The destination hostname or IP address.
    :param port: UDP port to send to.
    :param buffer: The payload buffer to send (bytes).
    :param frame_id: ID of the frame for telemetry.
    :return: The result of the send_data_with_headers function (1 for success, 0 for failure).
    """
    loop = asyncio.get_event_loop()
    # Run send_data_with_headers in an executor thread
    result = await loop.run_in_executor(
        executor,
        send_data_with_headers,
        host,  # Parameters to the send_data_with_headers function
        port,
        buffer,
        frame_id,
        object_id,
        total_objects,
    )
    return result

def send_data_with_headers(host, port, buffer, frame_id, object_id,total_objects):
    """
    Sends data over UDP with headers containing frame_id and packet sequence.
    :param host: The destination hostname or IP address.
    :param buffer: Data buffer to send (bytes).
    :param frame_id: ID of the current frame being sent.
    :return: 1 for success, 0 for failure.
    """
    sock = None

    try:
        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Resolve the host IP
        server_address = (host, port)

        # Calculate total packets
        total_packets = math.ceil(len(buffer) / MAX_PAYLOAD_SIZE)

        for i in range(total_packets):
            # Calculate payload size for this packet
            payload_size = min(len(buffer) - i * MAX_PAYLOAD_SIZE, MAX_PAYLOAD_SIZE)

            # Get the data chunk for this packet
            payload = buffer[i * MAX_PAYLOAD_SIZE: i * MAX_PAYLOAD_SIZE + payload_size]

            # Create the header
            header = PacketHeader(
                total_packets=total_packets,
                packet_index=i,
                frame_index=frame_id,
                object_index=object_id,
                total_objects=total_objects,
            )

            # Combine header and payload
            packet = header.pack() + payload

            # Send the packet
            sent = sock.sendto(packet, server_address)

            if sent == 0:
                print(f"Failed to send packet {i}")
                return 0

        # Success
        return 1

    except Exception as e:
        print(f"Error in send_data_with_headers: {e}")
        return 0

    finally:
        if sock:
            sock.close()


