# Host system time when the full payload was read from the socket.
std_msgs/Header header
# Firmware camera IDs: 0 = U2, 1 = U3.
uint8 camera_id
uint32 sequence
# V4L2 timestamp on the development board, in its own clock domain.
uint64 board_timestamp_us
# Host CLOCK_MONOTONIC: after sendall, after this payload, after the capture reply.
# Frames from one request share request_sent_monotonic_ns and capture_done_monotonic_ns.
uint64 request_sent_monotonic_ns
uint64 received_monotonic_ns
uint64 capture_done_monotonic_ns
# width is the number of active Bayer pixels. wire_width is the OWSF header value.
uint32 width
uint32 wire_width
uint32 height
uint32 stride
string fourcc
# Native BG10: four consecutive little-endian 10-bit samples per five bytes.
# Every row includes the original trailing padding up to stride bytes.
uint8[] data
