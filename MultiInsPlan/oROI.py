import spiceypy as spice


class ROI:
    def __init__(self, name, tw):
        self.ROI_name = name
        self.ROI_TW = tw

    def obsmakespan(self,t):

        if t >= 0 and t <= 11:
            return (1/11) * t + 0.9
        elif t > 11 and t <= 20:
            return (1/90) * (t - 20)**2 + 0.9


