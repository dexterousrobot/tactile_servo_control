import numpy as np
import matplotlib
matplotlib.use("TkAgg") # change backend to stop plt stealing focus
import matplotlib.pylab as plt

from cri.robot import quat2euler, euler2quat, inv_transform


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK; can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


class PlotContour2D:
    def __init__(self, 
                poses=[[-60,60], [-10,110]],
                workframe=[0, 0, 0, 0, 0, 90],
                position=[0, 400],
                r=[-1,1], 
                inv=-1
    ):
        self._fig = plt.figure('Contour 2d', figsize=(5, 5))
        self._fig.clear()
        self._ax = self._fig.add_subplot(111) 
        self._ax.set_aspect('equal', adjustable='box')
        
        self._ax.plot(poses[0], poses[1], ':w')  

        self.r = r[:2]/np.linalg.norm(r[:2])
        self.a, self.i = (workframe[5]-90)*np.pi/180, inv 

        self.v = [0, 0, 0, 0, 0, 0]

        move_figure(self._fig, *position)
        self._fig.show()

    def update(self, v):
        self.v = np.vstack([self.v, v])

        v_q = euler2quat([0, 0, 0, 0, 0, self.i*self.v[-1,5]+self.a], axes='rxyz')
        d_q = euler2quat([*self.r[::-self.i], 0, 0, 0, 0], axes='rxyz')
        d = 5*quat2euler(inv_transform(d_q, v_q), axes='rxyz')

        rv = np.zeros(np.shape(self.v))
        rv[:,0] = self.i * ( np.cos(self.a)*self.v[:,0] - np.sin(self.a)*self.v[:,1] )
        rv[:,1] = np.sin(self.a)*self.v[:,0] + np.cos(self.a)*self.v[:,1]
        
        self._ax.plot(rv[-2:,0], rv[-2:,1], '-r') 
        self._ax.plot(rv[-2:,0]+[d[0],-d[0]], rv[-2:,1]+[d[1],-d[1]], '-b', linewidth=0.5) 

        self._fig.canvas.flush_events()   # update the plot
        plt.pause(0.0001)


class PlotContour3D:
    def __init__(self, 
                workframe=[0, 0, 0, 0, 0, 180],
                stim_name=None,
                position=[0, 400],
                r=[-1,1]/np.sqrt(2),
                inv=1
                ):

        if stim_name=='saddle':
            limits = [[-70,70], [-10,130], [-30,30]]
        else:
            limits = [[-110,10], [-60,60], [-30,30]]

        self._fig = plt.figure('Contour 3d', figsize=(5, 5))
        self._fig.clear()
        self._fig.subplots_adjust(left=-0.1, right=1.1, bottom=-0.05, top=1.05)
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._ax.azim = workframe[5]
        self._ax.plot(limits[0], limits[1], limits[2], ':w')  

        self.r = r 
        self.inv = inv 
        self.v = [0, 0, 0, 0, 0, 0]

        move_figure(self._fig, *position)
        self._fig.show()

    def update(self, v):
        self.v = np.vstack([self.v, v])

        v_q = euler2quat([0, 0, 0, *self.v[-1,3:]], axes='rxyz')
        d_q = euler2quat([*self.r[::-1], 0, 0, 0, 0], axes='rxyz')
        w_q = euler2quat([0, 0, -1, 0, 0, 0], axes='rxyz')
        d = 5*quat2euler(inv_transform(d_q, v_q), axes='rxyz')
        w = 5*quat2euler(inv_transform(w_q, v_q), axes='rxyz')

        self._ax.plot(
            self.inv*self.v[-2:,0], -self.v[-2:,1], -self.v[-2:,2], 
            '-r') 
        self._ax.plot(
            self.inv*self.v[-2:,0]+[d[0],-d[0]], -self.v[-2:,1]-[d[1],-d[1]], -self.v[-2:,2]-[d[2],-d[2]], 
            '-b', linewidth=0.5) 
        self._ax.plot(
            self.inv*self.v[-2:,0]+[w[0],0], -self.v[-2:,1]-[w[1],0], -self.v[-2:,2]-[w[2],0], 
            '-g', linewidth=0.5) 

        self._fig.canvas.flush_events()   # update the plot
        plt.pause(0.0001)


if __name__ == '__main__':
    None
