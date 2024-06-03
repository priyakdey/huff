# huff.py

## Overview

`huff.py` is a compression tool, used compress/decompress files.

This project is an implementation of Huffman encoding and decoding as part of
the
[Huffman Coding Challenge from codingchallenge.fyi](https://codingchallenges.fyi/challenges/challenge-huffman).
Huffman coding is a lossless data compression algorithm that uses
variable-length codes to represent characters based on their frequencies.

## Usage

1. Make sure you have python3 installed (> 3.10)
2. To compress a file:

```console
$ ./huff.py compress <input_filepath>
```

3. This will generate a `input_filepath.huff` file.

4. To decompress a file: **TODO**

**Disclaimer**: This project is educational, recreational and experimental. Not
to be used for production environment.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE)
file for details.

## Acknowledgements

Special thanks to [patorjk.com](https://www.patorjk.com/software/taag/#p=display&f=Graffiti&t=Type%20Something%20) for the ASCII art generator.