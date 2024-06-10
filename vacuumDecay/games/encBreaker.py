from vacuumDecay import *
from arc4 import ARC4
import copy

class KnownPlaintextAndKeylen(State, ABC):
    def __init__(self, plaintext, ciphertext, keyLenBits, keyBits=None, turn=0, generation=0, playersNum=1, lastChange=None):
        if keyBits==None:
            keyBits = [0]*keyLenBits
        self.turn = turn
        self.generation = generation
        self.keyBits = keyBits
        self.keyLenBits = keyLenBits
        self.plaintext = plaintext
        self.ciphertext = ciphertext
        self.lastChange = lastChange
        self.decrypt = self._decrypt()
        self.score = self.getScore()

    def mutate(self, action):
        newKeyBits = copy.copy(self.keyBits)
        newKeyBits[action.data] = int(not newKeyBits[action.data])
        return XorKnownPlaintextAndKeylen(self.plaintext, self.ciphertext, self.keyLenBits, newKeyBits, generation=self.generation+1, lastChange = action.data)

    def getAvaibleActions(self):
        for i in range(self.keyLenBits):
            #if self.keyBits[i] == 0:
            if self.lastChange != i:
                yield Action(0, i)

    def getKey(self):
        s = ""
        for i in range(int(self.keyLenBits/8)):
            s += chr(int("".join([str(c) for c in self.keyBits[i*8:][:8]]),2))
        return s

    @abstractmethod
    def _decrypt(self):
        pass

    def checkWin(self):
        return self.decrypt == self.plaintext

    def getScore(self):
        diff = sum([bin(ord(a) ^ ord(b)).count("1") for a,b in zip(self.decrypt, self.plaintext)])
        return diff / (len(self.plaintext)*8)

    def __str__(self):
        return "{"+self.getKey()+"}["+self.decrypt+"]"

    def getTensor(self):
        return torch.tensor(self.keyBits + list(map(int, ''.join([bin(ord(i)).lstrip('0b').rjust(8,'0') for i in self.decrypt]))))

    def getModel(self):
        pass

    def getPriority(self, score):
        return self.score + (1/self.keyLenBits)*0.01*self.generation

class XorKnownPlaintextAndKeylen(KnownPlaintextAndKeylen):
    def _decrypt(self):
        return ''.join(chr(ord(a) ^ ord(b)) for a,b in zip(self.ciphertext, self.getKey()))

class RC4KnownPlayintextAndKeylen(KnownPlaintextAndKeylen):
    def _decrypt(self):
        rc4 = ARC4(self.getKey())
        return rc4.decrypt(self.ciphertext).decode("ascii")

if __name__=="__main__":
    vd = WeakSolver(RC4KnownPlaintextAndKeylen())

# TODO:
# - Should use bytes for everything (not array of ints / string)
