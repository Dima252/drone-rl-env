"""Measure the real node's control loop under different executors.

A separate process stands in for MAVROS, publishing 50 Hz telemetry on the real
topics with the real QoS, so the node runs its full active path.
"""

import multiprocessing as mp, os, sys, time
import numpy as np


def fake_mavros(stop):
    import rclpy
    from mavros_msgs.msg import State
    from nav_msgs.msg import Odometry
    from rcl_interfaces.msg import ParameterType, ParameterValue
    from rcl_interfaces.srv import GetParameters
    from rclpy.qos import qos_profile_sensor_data as Q
    from sensor_msgs.msg import Imu

    rclpy.init()
    n = rclpy.create_node("fake_mavros")
    st = n.create_publisher(State, "/mavros/state", Q)
    imu = n.create_publisher(Imu, "/mavros/imu/data", Q)
    od = n.create_publisher(Odometry, "/mavros/local_position/odom", Q)

    def params(req, res):
        res.values = [ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=8)]
        return res

    n.create_service(GetParameters, "/mavros/param/get_parameters", params)
    t0 = time.monotonic()
    count = [0]

    def tick():
        stamp = n.get_clock().now().to_msg()
        i = Imu()
        i.header.stamp = stamp
        i.orientation.w = 1.0
        imu.publish(i)
        o = Odometry()
        o.header.stamp = stamp
        o.pose.pose.position.z = 15.0
        o.pose.pose.orientation.w = 1.0
        od.publish(o)
        count[0] += 1
        if count[0] % 10 == 0:
            s = State()
            s.connected = True
            s.armed = True
            s.mode = "LOITER" if time.monotonic() - t0 < 5 else "GUIDED"
            st.publish(s)

    n.create_timer(0.02, tick)
    while not stop.is_set():
        rclpy.spin_once(n, timeout_sec=0.05)


def run(label, executor_name, seconds=20):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor

    sys.path.insert(0, "/ws")
    import ros2_inference_node as rn

    ticks, active = [], []

    class Timed(rn.InferenceNode):
        def on_control_tick(self):
            t = time.monotonic()
            ticks.append(t)
            super().on_control_tick()
            active.append((t, time.monotonic(), self.status.value))

    rclpy.init()
    node = Timed.__new__(Timed)
    sys.argv = [
        "x",
        "--ros-args",
        "-p",
        "model_path:=/policy/model.onnx",
        "-p",
        "policy_meta_path:=/policy/policy_meta.npz",
        "-p",
        "max_position_m:=100.0",
    ]
    rclpy.shutdown()
    rclpy.init(args=sys.argv)
    Timed.__init__(node)
    ex = (
        SingleThreadedExecutor()
        if executor_name == "single"
        else MultiThreadedExecutor(num_threads=3)
    )
    ex.add_node(node)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        ex.spin_once(timeout_sec=0.1)
    on = [a for a in active if a[2] == "active"]
    gaps = np.diff([a[0] for a in on]) * 1e3 if len(on) > 2 else np.array([0.0])
    work = np.array([a[1] - a[0] for a in on]) * 1e3 if on else np.array([0.0])
    print(
        f"{label:28s} active ticks {len(on):4d}  rate {1000/gaps.mean():5.1f} Hz  period std {gaps.std():5.2f}  p99 {np.percentile(gaps,99):6.1f}  max {gaps.max():6.1f} ms  late {int((gaps>30).sum()):3d}  tick work p99 {np.percentile(work,99):.2f} ms",
        flush=True,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    for label, name in (("MultiThreaded", "multi"), ("SingleThreaded (the node)", "single")):
        stop = mp.Event()
        f = mp.Process(target=fake_mavros, args=(stop,))
        f.start()
        time.sleep(1)
        w = mp.Process(target=run, args=(label, name))
        w.start()
        w.join(60)
        stop.set()
        f.join(5)
        f.terminate()
        if w.is_alive():
            w.terminate()
