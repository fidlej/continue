
def to_binseq(bytes):
    seq = []
    for byte in bytes:
        value = ord(byte)
        for i in xrange(7, -1, -1):
            one = value & 2**i
            seq.append("1" if one else "0")

    return "".join(seq)


def to_bytes(bits):
    assert len(bits) % 8 == 0
    bytes = []
    index = 0
    while index < len(bits):
        value = 0
        for i in xrange(7, -1, -1):
            if bits[index + (7 - i)] == "1":
                value |= 2**i

        bytes.append(chr(value))
        index += 8

    return "".join(bytes)


