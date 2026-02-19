import random

import numpy
import tcp.comm as comm

class Tpc_env():
    def connect(self):
        self.cl = comm.Connect()
        print("WE ARE ONLINE")
        self.time = 0



    def step(self,actions):
        """
        performs a stepp in the env
        :param actions:
        :return:
        """

        data = [0] + actions.tolist()

        
        recived = comm.get_and_send_data(data,self.cl)

        revard = recived[0]

        term = recived[1] == -1

        obs = recived[2:]

        self.time += 1
        return numpy.array(obs),revard,term,self.time > 200,None

    def reset (self):
        data = [-1] + ([0]*9)

        recived = comm.get_and_send_data(data,self.cl)

        revard = recived[0]

        done = recived[1] == -1

        obs = recived[2:]
        self.time = 0

        return obs , None

    def close(self):
        self.cl.close()