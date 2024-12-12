class Instrument:
    def __init__(self, type, *args):
        self.type = type

        if type == 'CAMERA':
            self.ifov = args[0]
            self.npix = args[1]
            self.imageRate = args[2]
            self.safetyFactor = args[3]

        if type == 'RADAR':
            None

