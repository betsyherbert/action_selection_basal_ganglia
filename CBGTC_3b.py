# Simulation of whole cortico-baso-thalamo-cortical loop (CBGTC)

import math
import random
import numpy
import numpy.random
import basalganglia
import thalamusFC


# -------------------------------------------
class CBGTC:
    # -----------------------------------------
    def __init__(self, saliences, model='CBG', NbChannels=5):
        self.model = model
        self.NbChannels = NbChannels
        self.BG = basalganglia.BasalGanglia(model, NbChannels)
        self.THFC = thalamusFC.thalamusFC(model, NbChannels)
        self.restInhibition = self.getInhibRest(0.001)
        
        self.saliences = saliences

    def stepCompute(self, dt, saliences):
        inhibs = self.BG.readGPi()
        FCout = self.THFC.readFC()
        self.BG.stepCompute(dt, saliences, FCout)
        self.THFC.stepCompute(dt, saliences, inhibs)

    def nbStepsCompute(self, dt, NbSteps, saliences):

        for t in range(NbSteps):
            self.stepCompute(dt, saliences)
        return self.BG.readGPi()

    def getInhibRest(self, dt):
        saliences = numpy.zeros((self.NbChannels))
        inh = self.nbStepsCompute(dt, 1000, saliences)
        return inh[0]

    # --------------------------------------------------------
    # simulates the selection test 
    def selectionTest(self, dt, saliences):

        inhibs = self.nbStepsCompute(dt, 2000, saliences)

        selected = 0

        if ((inhibs[0] < self.restInhibition) and (inhibs[1] >= self.restInhibition)) or (
                (inhibs[0] >= self.restInhibition) and (inhibs[1] < self.restInhibition)):
            selected = 1

        elif (inhibs[0] < self.restInhibition) and (inhibs[1] < self.restInhibition):
            selected = 2

        return selected


# ---------------------------

def main(saliences):
    
    dt = 0.001

    myCBGTC = CBGTC(saliences)

    selected = myCBGTC.selectionTest(dt, saliences)
    
    return selected


# ---------------------------

if __name__ == '__main__':
    # Import Psyco if available
    try:
        import psyco

        psyco.log()
        psyco.profile()
        psyco.full()
    except ImportError:
        print('Psyco not available.')  # project died in 2012 :(
    main()


