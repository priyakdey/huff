#!/usr/bin/env python3


#####################################################################################
#
#                        _    _        __  __                                       #
#                       | |  | |      / _|/ _|                                      #
#                       | |__| |_   _| |_| |_                                       #
#                       |  __  | | | |  _|  _|                                      #
#                       | |  | | |_| | | | |                                        #
#                       |_|  |_|\__,_|_| |_|                                        #
#                                                                                   #
# https://www.patorjk.com/software/taag/#p=display&f=Graffiti&t=Type%20Something%20 #
#                                                                                   #
#####################################################################################


from abc import ABC, abstractmethod
import struct
import sys
from typing import Dict, List, Tuple


class HuffmanNode(ABC):
    """Abstract base class for Huffman tree nodes."""

    @property
    @abstractmethod
    def codepoint(self) -> int:
        """Returns the codepoint of the node"""
        pass

    @property
    @abstractmethod
    def freq(self) -> int:
        """Returns the frequency of the codepoint"""
        pass

    def serialize(self) -> bytes:
        """
        Serializes the Huffman tree into bytes.

        Args:
            root (HuffmanNode): The root of the Huffman tree.

        Returns:
            bytes: The serialized Huffman tree.
        """
        nodes: List[Tuple[int, int, int, int]] = []

        def serialize(node: HuffmanNode) -> int:
            nonlocal nodes

            if isinstance(node, HuffmanLeafNode):
                nodes.append((node.codepoint, node.freq, -1, -1))
            # without this, the compiler cannot infer the type, even though we have 2 types only !!!
            elif isinstance(node, HuffmanNonLeafNode):
                left_index = serialize(node.left)
                right_index = serialize(node.right)
                nodes.append((node.codepoint, node.freq, left_index, right_index))

            return len(nodes) - 1

        serialize(self)

        # ref: serialize.txt
        packed_tree = [
            struct.pack("<iiii", cp, freq, left, right)
            for cp, freq, left, right in nodes
        ]
        buf = b"".join(packed_tree)

        length = len(buf)
        buf = struct.pack("<I", length) + buf

        return buf

    def __lt__(self, other: "HuffmanNode") -> bool:
        """Compares this node with another node based on frequency and codepoint."""
        if self.freq == other.freq:
            return self.codepoint <= other.codepoint

        return self.freq < other.freq


class HuffmanLeafNode(HuffmanNode):
    """Represents a leaf node in the Huffman tree."""

    def __init__(self, codepoint: int, freq: int):
        """
        Initializes a HuffmanLeafNode.

        Args:
            codepoint (int): The codepoint of the character.
            freq (int): The frequency of the character.
        """
        self._codepoint = codepoint
        self._freq = freq
        self.left = None
        self.right = None

    @property
    def codepoint(self) -> int:
        return self._codepoint

    @property
    def freq(self) -> int:
        return self._freq


class HuffmanNonLeafNode(HuffmanNode):
    """Represents a non-leaf node in the Huffman tree."""

    def __init__(self, left: HuffmanNode, right: HuffmanNode):
        """
        Initializes a HuffmanNonLeafNode.

        Args:
            left (HuffmanNode): The left child node.
            right (HuffmanNode): The right child node.
        """
        self._codepoint = 0x00
        self._freq = left.freq + right.freq
        self.left = left
        self.right = right

    @property
    def codepoint(self) -> int:
        return self._codepoint

    @property
    def freq(self) -> int:
        return self._freq


class PriorityQueue:
    """A priority queue implementation for Huffman nodes."""

    def __init__(self, capacity: int):
        """
        Initializes the priority queue.

        Args:
            capacity (int): The capacity of the priority queue.
        """
        self.data: List[HuffmanNode] = [HuffmanLeafNode(0x00, -1)] * capacity
        self.length: int = 0
        self.capacity: int = capacity

    def push(self, node: HuffmanNode) -> None:
        """
        Pushes a node into the priority queue.

        Args:
            node (HuffmanNode): The node to be added.
        """
        self.data[self.length] = node
        self.length += 1

        curr_index = self.length - 1

        while curr_index >= 0:
            parent_index = self.get_parent_index(curr_index)

            curr_node = self.data[curr_index]
            parent_node = self.data[parent_index]

            if curr_node < parent_node:
                self.swap(curr_index, parent_index)
            else:
                break

            curr_index = parent_index

    def pop(self) -> HuffmanNode:
        """
        Pops the highest priority node from the queue.

        Returns:
            HuffmanNode: The highest priority node.
        """
        root = self.data[0]
        self.swap(0, self.length - 1)

        self.length -= 1

        curr_index = 0

        while curr_index < self.length:
            left_index, right_index = self.get_child_index(curr_index)

            if left_index >= self.length:
                break

            swap_index = left_index
            if (
                right_index < self.length
                and self.data[right_index] < self.data[left_index]
            ):
                swap_index = right_index

            curr_node = self.data[curr_index]
            swap_node = self.data[swap_index]
            if swap_node < curr_node:
                self.swap(curr_index, swap_index)
            else:
                break

            curr_index = swap_index

        return root

    def swap(self, i: int, j: int) -> None:
        """
        Swaps two nodes in the queue.

        Args:
            i (int): The index of the first node.
            j (int): The index of the second node.
        """
        self.data[i], self.data[j] = self.data[j], self.data[i]

    def get_parent_index(self, index: int) -> int:
        """
        Gets the parent index of a node.

        Args:
            index (int): The index of the node.

        Returns:
            int: The parent index.
        """
        return (index - 1) // 2

    def get_child_index(self, index: int) -> Tuple[int, int]:
        """
        Gets the child indices of a node.

        Args:
            index (int): The index of the node.

        Returns:
            Tuple[int, int]: The indices of the left and right children.
        """
        return 2 * index + 1, 2 * index + 2

    def __len__(self) -> int:
        return self.length


def encode_content(
    codepoints: List[int], code_prefix_table: Dict[int, Tuple[int, int]]
) -> bytes:
    """
    Encodes the content using the code prefix table.

    Args:
        codepoints (List[int]): The list of codepoints to be encoded.
        code_prefix_table (Dict[int, Tuple[int, int]]): The code prefix table.

    Returns:
        bytes: The encoded content.
    """
    buffer = bytearray()
    current_byte = 0
    current_bit_length = 0

    # this took me 3 days to work on... wow, I am stupid :(
    for codepoint in codepoints:
        code, length = code_prefix_table[codepoint]
        while length > 0:
            remaining_bits = 8 - current_bit_length
            if length <= remaining_bits:
                current_byte = (current_byte << length) | code
                current_bit_length += length
                length = 0
            else:
                current_byte = (current_byte << remaining_bits) | (
                    code >> (length - remaining_bits)
                )
                code &= (1 << (length - remaining_bits)) - 1
                length -= remaining_bits
                current_bit_length = 8

            if current_bit_length == 8:
                buffer.append(current_byte)
                current_byte = 0
                current_bit_length = 0

    if current_bit_length > 0:
        buffer.append(current_byte << (8 - current_bit_length))

    return bytes(buffer)


def generate_code_prefix_table(root: HuffmanNode) -> Dict[int, Tuple[int, int]]:
    """
    Generates a code prefix table from the Huffman tree.

    Args:
        root (HuffmanNode): The root of the Huffman tree.

    Returns:
        Dict[int, Tuple[int, int]]: The code prefix table.
    """
    code_prefix_table: Dict[int, Tuple[int, int]] = {}

    def dfs(node: HuffmanNode, code: int, length: int):
        nonlocal code_prefix_table

        if isinstance(node, HuffmanLeafNode):
            code_prefix_table[node.codepoint] = (code, length)
        elif isinstance(node, HuffmanNonLeafNode):
            dfs(node.left, code << 1, length + 1)
            dfs(node.right, (code << 1) | 1, length + 1)

    dfs(root, 0, 0)
    return code_prefix_table


def generate_huffman_tree(freq_table: Dict[int, int]) -> HuffmanNode:
    """
    Generates a Huffman tree from a frequency table.

    Args:
        freq_table (Dict[int, int]): The frequency table.

    Returns:
        HuffmanNode: The root of the generated Huffman tree.
    """
    pq = PriorityQueue(len(freq_table))

    for codepoint, freq in freq_table.items():
        pq.push(HuffmanLeafNode(codepoint, freq))

    while len(pq) > 1:
        left = pq.pop()
        right = pq.pop()
        pq.push(HuffmanNonLeafNode(left, right))

    return pq.pop()


def generate_codepoint_freq_table(codepoints: List[int]) -> Dict[int, int]:
    """
    Generates a frequency table from a list of codepoints.

    Args:
        codepoints (List[int]): The list of codepoints.

    Returns:
        Dict[int, int]: The frequency table.
    """
    freq_table: Dict[int, int] = {}
    for codepoint in codepoints:
        freq_table[codepoint] = freq_table.get(codepoint, 0) + 1
    return freq_table


def decode_bytes_to_codepoints(bytes_buffer: List[int]) -> List[int]:
    """
    Decodes bytes to codepoints.

    Args:
        bytes_buffer (List[int]): The bytes buffer to decode.

    Returns:
        List[int]: The list of decoded codepoints.
    """
    cursor = 0
    codepoints: List[int] = []

    while cursor < len(bytes_buffer):
        lb = bytes_buffer[cursor]

        # ref: https://www.freecodecamp.org/news/what-is-utf-8-character-encoding/
        # ref: https://github.com/golang/go/blob/master/src/unicode/utf8/utf8.go#L151
        # Easier way is to port the thing into python, but it has BSD-3 license,
        # which I dont understand shit about including!! So just stole some "bits".
        if lb <= 197:
            codepoint = lb
            length = 1
        elif lb >= 194 and lb <= 223:
            codepoint = (lb & 0x1F) << 6 | (bytes_buffer[cursor + 1] & 0x37)
            length = 2
        elif lb >= 224 and lb <= 239:
            codepoint = (
                (lb & 0x0F) << 12
                | (bytes_buffer[cursor + 1] & 0x37) << 6
                | (bytes_buffer[cursor + 2] & 0x3F)
            )
            length = 3
        else:
            codepoint = (
                (lb & 0x07) << 18
                | (bytes_buffer[cursor + 1] & 0x3F) << 12
                | (bytes_buffer[cursor + 2] & 0x3F) << 6
                | (bytes_buffer[cursor + 3] & 0x3F)
            )
            length = 4

        codepoints.append(codepoint)
        cursor += length

    return codepoints


def read_file(filepath: str, chunk_size: int) -> List[int]:
    """
    Reads a file and returns its content as a list of bytes.

    Args:
        filepath (str): The path to the file.
        chunk_size (int): The chunk size for reading the file.

    Returns:
        List[int]: The content of the file as a list of bytes.
    """
    buffer: List[int] = []
    with open(filepath, "rb") as fd:
        while chunk := fd.read(chunk_size):
            buffer.extend(chunk)
    return buffer


def compress(filepath: str) -> None:
    """
    Compresses a file using Huffman encoding.

    Args:
        filepath (str): The path to the file to compress.
    """
    print(f"compressing file {filepath}...")
    content = read_file(filepath, 4096)
    codepoints = decode_bytes_to_codepoints(content)
    codepoint_freq_table = generate_codepoint_freq_table(codepoints)
    huffman_tree = generate_huffman_tree(codepoint_freq_table)
    code_prefix_table = generate_code_prefix_table(huffman_tree)

    serialized_tree = huffman_tree.serialize()
    encoded_content = encode_content(codepoints, code_prefix_table)

    # append encoded_content to end of serialized tree and flush to file
    out_filepath = filepath + ".huff"

    with open(out_filepath, "wb") as f:
        f.write(serialized_tree + encoded_content)
        f.flush()

    print(f"compressed data written to {out_filepath}")


def print_usage(program: str) -> None:
    """
    Prints the usage information for the program.

    Args:
        program (str): The name of the program.
    """
    print(f"\nUsage: {program} <compress|decompress> <input_filepath>")
    print("\nSubcommands:")
    print("\tcompress\t\tcompress the input file")
    print("\tdecompress\t\tdecompress the input file")


if __name__ == "__main__":
    program, *args = sys.argv

    if len(args) < 1:
        print("ERROR: no subcommand provided", file=sys.stderr)
        print_usage(program)
        exit(1)

    subcmd, *args = args

    match subcmd:
        case "compress":
            if len(args) < 1:
                print("ERROR: no input file provided", file=sys.stderr)
                print_usage(program)
                exit(1)

            input_filepath = args[0]
            compress(input_filepath)

        case "decompress":
            raise NotImplemented("decompress is not implemented")

        case _:
            print("ERROR: unknown subcommand", file=sys.stderr)
            print_usage(program)
            exit(1)
