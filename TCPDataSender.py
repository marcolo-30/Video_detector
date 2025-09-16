import socket
import struct
import time


class PacketHeader:
    """
    A class to handle TCP header creation and struct packing/unpacking.
    """

    def __init__(self, total_packets, packet_index, frame_index, object_index, total_objects,size):
        self.total_packets = total_packets
        self.packet_index = packet_index
        self.frame_index = frame_index
        self.object_index = object_index
        self.total_objects = total_objects
        self.object_size = size

    def pack(self):
        # Pack the header as a binary struct
        return struct.pack('<IIIIII', self.total_packets, self.packet_index,
                           self.frame_index, self.object_index, self.total_objects,self.object_size)


class TCPClient:
    """
    A TCP client class to establish a persistent connection for sending data with headers.
    """

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None

    def __del__(self):
        if self.sock is not None:
            self.sock.close()

    def connect(self):
        """
        Establishes a persistent TCP connection to the server.
        """
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"Error connecting to server: {e}")
            self.sock = None

    def reconnect(self, max_retries=5, retry_interval=3):
        """
        Attempts to reconnect to the server.
        :param max_retries: Maximum number of reconnection attempts.
        :param retry_interval: Time in seconds to wait between retries.
        :return: True if reconnected successfully, False otherwise.
        """
        self.close()  # Ensure existing socket is closed before reconnecting
        retries = 0
        while retries < max_retries:
            try:
                print(f"Attempting to reconnect to {self.host}:{self.port} (Attempt {retries + 1}/{max_retries})...")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                print("Reconnected successfully.")
                return True
            except Exception as e:
                print(f"Reconnection attempt {retries + 1} failed: {e}")
                retries += 1
                time.sleep(retry_interval)  # Wait before another attempt

        print(f"Failed to reconnect after {max_retries} attempts.")
        return False

    def send_data(self, buffer, frame_id, object_id, total_objects):
        """
        Sends data over the persistent TCP connection with headers.
        :param buffer: Data buffer to send (bytes).
        :param frame_id: ID of the current frame being sent.
        :param object_id: ID of the object being sent.
        :param total_objects: Total number of objects in the transmission.
        :return: 1 for success, 0 for failure.
        """
        if self.sock is None:
            print("Error: No active connection. Trying to reconnect...")
            if not self.reconnect():
                return 0

        try:
            # Create the header
            total_packets = 1  # Just one logical packet for TCP
            header = PacketHeader(
                total_packets=total_packets,
                packet_index=0,  # Single packet
                frame_index=frame_id,
                object_index=object_id,
                total_objects=total_objects,
                size=len(buffer)
            )

            # Combine header and full payload
            packet = header.pack() + buffer

            # Send the packet
            self.sock.sendall(packet)
            print(f"Data sent successfully, total size: {len(packet)} bytes.")
            return 1

        except (BrokenPipeError, ConnectionError) as e:
            print(f"Connection error while sending data: {e}. Trying to reconnect...")
            if self.reconnect():
                # Retry sending after successfully reconnecting
                print("Reconnected. Retrying data transmission...")
                try:
                    self.sock.sendall(packet)
                    print(f"Data resent successfully, total size: {len(packet)} bytes.")
                    return 1
                except Exception as e:
                    print(f"Error resending data after reconnect: {e}")
            return 0
        except Exception as e:
            print(f"Unexpected error sending data: {e}")
            return 0

    def close(self):
        """
        Closes the TCP connection.
        """
        if self.sock:
            try:
                self.sock.close()
                self.sock = None
                print("Connection closed.")
            except Exception as e:
                print(f"Error closing connection: {e}")

