import codecs
import io

# Buffer size in bytes
BUFFER = 100000000


# def readfile(file_obj):
#   # Assume postion of seek is 0.
#   current_offset = file_obj.tell()
#   text = StringIO.StringIO(file_obj.read())
#   new_line_offsets = []
#   while True:
#     x = text.read(1)
#     if  x == "":
#       break
#     for char in x:
#       if "\n" == char:
#         new_line_offsets.append(text.tell())
#   text.seek(0)
#   for idx, offset in enumerate(new_line_offsets):
#     prev = 0 if  idx == 0 else new_line_offsets[idx-1]
#     to_read = new_line_offsets[idx] - prev
#     yield (text.read(to_read), offset)


# #V2 x10
# def get_string_offset(string, bytes_read):
#   offsets = []
#   start_index = 0
#   for idx, char in enumerate(string):
#     if char == "\n":
#       offsets.append(len(bytes(string[0:idx+1])) + bytes_read)
#       start_index = idx
#   return offsets, ""

# def readfile(file_obj):
#   # Assume postion of seek is 0.
#   current_offset = file_obj.tell()
#   text = StringIO.StringIO(file_obj.read())
#   new_line_offsets = []
#   bytes_read = 0
#   bytes_to_read = 1000
#   while True:
#     x = text.read(bytes_to_read)
#     if x == "":
#       break
#     offsets, remainder = get_string_offset(x, bytes_read)
#     bytes_read += bytes_to_read
#     new_line_offsets.extend(offsets)
#   text.seek(0)
#   for idx, offset in enumerate(new_line_offsets):
#     prev = 0 if  idx == 0 else new_line_offsets[idx-1]
#     to_read = new_line_offsets[idx] - prev
#     yield (text.read(to_read), offset)

# V3 x20

def get_string_offset2(string, bytes_read, file_obj):
    # print "BYTES READ"
    # print string
    # print bytes_read
    # print "END BYTES READ"
    # print string
    offsets = []
    last_newline_index = 0
    loop_bytes_read = 0
    for idx, char in enumerate(string):
        if char == "\n":
            try:
                offset_value = loop_bytes_read + bytes_read
            except:
                import pdb;
                pdb.set_trace()
            offsets.append((string[last_newline_index:idx + 1], offset_value))
            loop_bytes_read = len(string[0:idx + 1].encode('utf-8'))
            last_newline_index = idx + 1
            if False:
                orig = file_obj.tell()
                try:
                    file_obj.seek(offsets[-1][1])
                except:
                    import pdb;
                    pdb.set_trace()
                expected = file_obj.read(1)
                real = offsets[-1][0][0]
                orig = file_obj.seek(orig)
                try:
                    assert real == expected
                except:
                    import pdb;
                    pdb.set_trace()
    return offsets, string[last_newline_index:]


def ilc_readfile(file_obj):
    stop = False
    # Assume postion of seek is 0.
    current_offset = file_obj.tell()
    text = file_obj  # file_obj.read()
    # if data == "":
    #   stop = True
    # else:
    #   text = StringIO.StringIO(data)
    new_line_offsets = []
    bytes_read = 0
    bytes_to_read = 32
    remainder = ""
    counter = 0
    # Per gestire lunghezze diverse caratteri unicode
    while stop == False:
        counter = counter + 1
        before_read = text.tell()
        try:
            x = text.read(bytes_to_read)
        except:
            text.seek(before_read - 1)
            try:
                x = text.read(bytes_to_read)
            except:
                text.seek(before_read - 2)
                try:
                    x = text.read(bytes_to_read)
                except:
                    text.seek(before_read - 3)
                    x = text.read(bytes_to_read)
        after_read = text.tell()
        real_bytes_read = after_read - before_read
        if x == "":
            break
        offsets, remainder = get_string_offset2(remainder + x,
                                                bytes_read - len(remainder.encode('utf-8')),
                                                file_obj)
        current_offset = file_obj.tell()
        for y in offsets:
            yield y
        current_offset = file_obj.seek(current_offset)
        bytes_read = bytes_read + real_bytes_read
        new_line_offsets.extend(offsets)
# if __name__ == '__main__':
#   for x in readfile(open("pippo_big")):
#     print x


# def get_string_offset2(string, bytes_read):
#   offsets = []
#   last_newline_index = 0
#   for idx, char in enumerate(string):
#     if char == "\n":
#       offset = len(bytes(string[0:idx+1])) + bytes_read
#       #offsets.append(len(bytes(string[0:idx+1])) + bytes_read)
#       offsets.append((string[last_newline_index:idx+1], offset))
#       last_newline_index = idx + 1
#   return offsets, string[last_newline_index:]

# def ilc_readline(file_obj):
#   # Assume postion of seek is 0.
#   original_offset = file_obj.tell()
#   text = StringIO.StringIO(file_obj.read())
#   new_line_offsets = []
#   bytes_read = 0
#   bytes_to_read = 100
#   remainder = ""
#   while True:
#     x = text.read(bytes_to_read)
#     if x == "":
#       break
#     #print remainder
#     offsets, remainder = get_string_offset2(remainder + x, bytes_read -len(bytes(remainder)))
#     if len(offsets) > 0:
#       file_obj.seek(original_offset + offsets[0][1])
#       return offsets[0][0], original_offset + offsets[0][1]
#     bytes_read = bytes_read + bytes_to_read
#     new_line_offsets.extend(offsets)

# if __name__ == '__main__':
#   f = open("pippo_big")
#   while True:
#     res = ilc_readline(f)
#     #print res
#     if res == None:
#       break
