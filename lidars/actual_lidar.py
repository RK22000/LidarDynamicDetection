import numpy as np
import time
import traceback

from LidarClass import _Lidar
from rplidar import RPLidar, RPLidarException
from threading import Thread


class ActualLidar(_Lidar):
    def __init__(self, port) -> None:
        self.serial_port = port
        self.measures = []

    def connect_to_RPLidar(
        self, max_attempts=3, wait_seconds=1, verbose_attempts=True
    ) -> bool:
        for _ in range(max_attempts):
            print("hi there")
            try:
                print(f"trying to connect on port {self.serial_port}")
                self.lidar = RPLidar(self.serial_port, baudrate=256000)
                # self.lidar = RPLidar(self.serial_port, baudrate=115200)
                self.lidar_iter = self.lidar.iter_scans(max_buf_meas=10_000)
                next(self.lidar_iter)
                return True
            except RPLidarException as e:
                self.lidar.disconnect()
                continue
            # except PortNotOpenError:
            #     print(f"Actual Lidar Not connected: port {self.serial_port}")
            except:
                print("=================================================")
                print("Lidar Service Failed before lidar could start")
                print(self.serial_port)
                print(traceback.format_exc())
            time.sleep(wait_seconds)
        print("failed")
        return False

    def connect(self, max_attempts=3, wait_seconds=1, verbose_attempts=False) -> bool:
        if self.serial_port is None: return False
        if not self.connect_to_RPLidar(max_attempts, wait_seconds, verbose_attempts):
            return False
        self.stay_connected = True

        def look():
            while self.stay_connected:
                self.measures = [(q,-a%360,d) for q,a,d in next(self.lidar_iter)]

        self.thread = Thread(target=look)
        self.thread.start()
        return True

    def disconnect(self):
        self.stay_connected = False
        self.thread.join()
        lidar = self.lidar
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()

    def get_measures(self, neighbors=0, distance=0):
        # if neighbors is None:
        return np.array(self.measures)
        points = np.array(self.measures)
        distances = points[:, 2]
        nxt = [*distances[1:],distances[0]]
        prv = [distances[-1], *distances[:-1]]
        front_distance = abs(distances-nxt)
        back_distance = abs(distances-prv)
        front_ok = front_distance < distances
        back_ok = back_distance < distances
        idx = np.logical_or(front_ok, back_ok)
        # print(sum(idx)/len(idx))
        return points[idx]
        


if __name__ == "__main__":
    port = "COM8"
    print(f"Using port: {port}")
    # exit()
    import time

    lidar = ActualLidar(port)
    lidar.test_Lidar()
