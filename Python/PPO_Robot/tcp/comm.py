import random
import socket

import tcp.decoder as decoder

packet_size = (2+40)*4

# where first 2 are rev + term , other are obs

def Connect():
    # create an INET, STREAMing socket
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # bind the socket to a public host, and a well-known port
    serversocket.bind(("localhost", 10003))
    # become a server socket
    serversocket.listen(5)
    print("waiting for a client")

    (clientsocket, address) = serversocket.accept()
    print(clientsocket)
    serversocket.close()
    return clientsocket


def get_and_send_data(data, clientsocket):
    clientsocket.send(decoder.pack_float32_array(data))



    while True:
        buf = clientsocket.recv(packet_size)
       
        if buf != 0:
            return decoder.unpack_float32_array(buf)






