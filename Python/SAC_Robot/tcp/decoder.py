import struct
import array


def pack_float32_array(data_array):
    """
    Packs a list or array of floats into a bytes object.

    The format is a sequence of 4-byte floats. The length is not included.

    Args:
        data_array (list or array.array): A list or array of floats to pack.

    Returns:
        bytes: The packed binary data.
    """
    # Ensure the input is an array of type 'f' (float32)
    if not isinstance(data_array, array.array):
        try:
            # Create an array of single-precision floats
            data_array = array.array('f', data_array)
        except TypeError:
            raise ValueError("Input data must be a sequence of numbers.")

    # Get the number of floats in the array
    num_floats = len(data_array)

    # Pack the floats themselves.
    # '!' is used for network byte order (big-endian), which is standard for network protocols.
    # The format string will be something like '!ff...' with one 'f' for each float.
    format_string = f'{num_floats}f'

    # Pack the data
    packed_data = struct.pack(format_string, *data_array)

    return packed_data


def unpack_float32_array(packed_data):
    """
    Unpacks a bytes object into a list of floats.

    This function assumes the entire byte string is a sequence of 4-byte floats.

    Args:
        packed_data (bytes): The binary data received from the socket.

    Returns:
        list: A list of the unpacked float values.
    """
    if len(packed_data) % 4 != 0:
        raise ValueError("Invalid data size: length must be a multiple of 4.")

    # Calculate the number of floats based on the data length
    num_floats = len(packed_data) // 4

    # Define the format string to unpack all floats at once
    format_string = f'{num_floats}f'

    # Unpack the data
    unpacked_tuple = struct.unpack(format_string, packed_data)

    return list(unpacked_tuple)
