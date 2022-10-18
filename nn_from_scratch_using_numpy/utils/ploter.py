from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter

class Ploter:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.pred = []

    def update_pred(self, pred):
        self.pred.append(pred)

    def show(self, out_file_name = 'out.gif', show = True):
        fig = plt.figure()
        plt.scatter(self.x, self.y)
        # plot  坐标必须为有序
        pred_fig, = plt.plot([], [], 'm')
        
        writer = PillowWriter(fps = 10)
        with writer.saving(fig, out_file_name, 100):
            for i in range(len(self.pred)):
                pred_fig.set_data(self.x, self.pred[i])
                writer.grab_frame()
                if show == True:
                    plt.pause(0.01)
        if show == True:
            plt.show()
