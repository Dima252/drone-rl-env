#!/usr/bin/env python3
"""Fly ros2_inference_node.py on real MAVROS and real ArduCopter SITL.

A scripted pilot flies the vehicle and hands control over by selecting GUIDED.
Attitude is decoded here with rotation matrices, independently of the node.
"""

import math, os, shutil, signal, socket, struct, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
import rclpy
from mavros_msgs.msg import AttitudeTarget, State, VfrHud
from mavros_msgs.srv import CommandBool, CommandTOL, ParamSetV2, SetMode, StreamRate
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as QOS
from sensor_msgs.msg import Imu

sys.path.insert(0, "/test")
import make_const

OUT = Path("/out")
TEST = Path("/test")
POLICY = Path("/tmp/policy")
USER_POLICY = Path("/policy")
POLICY_SECONDS = float(os.environ.get("POLICY_SECONDS", "10"))
HEADING_DEG = 60.0
ALT = 15.0
RESULTS = []
T = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])
F = np.diag([1.0, -1.0, -1.0])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    log(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def q2R(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def euler_ned(msg_q):
    """Roll, pitch, yaw of a ROS ENU/FLU quaternion, as an aircraft would state them."""
    R = T @ q2R([msg_q.w, msg_q.x, msg_q.y, msg_q.z]) @ F
    return (
        math.atan2(R[2, 1], R[2, 2]),
        -math.asin(max(-1, min(1, R[2, 0]))),
        math.atan2(R[1, 0], R[0, 0]),
    )


class Pilot(Node):
    def __init__(self):
        super().__init__("sitl_pilot")
        self.state = None
        self.imu_t = []
        self.odom_t = []
        self.set_t = []
        self.imu = None
        self.odom = None
        self.setpoint = None
        self.echo = None
        self.echo_t = []
        self.create_subscription(State, "/mavros/state", self.on_state, QOS)
        self.create_subscription(Imu, "/mavros/imu/data", self.on_imu, QOS)
        self.create_subscription(Odometry, "/mavros/local_position/odom", self.on_odom, QOS)
        self.create_subscription(AttitudeTarget, "/mavros/setpoint_raw/attitude", self.on_set, QOS)
        self.create_subscription(
            AttitudeTarget, "/mavros/setpoint_raw/target_attitude", self.on_echo, QOS
        )
        self.throttle = []
        self.create_subscription(
            VfrHud, "/mavros/vfr_hud", lambda m: self.throttle.append(m.throttle), QOS
        )
        self.mode_cli = self.create_client(SetMode, "/mavros/set_mode")
        self.arm_cli = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.tol_cli = self.create_client(CommandTOL, "/mavros/cmd/takeoff")
        self.pset_cli = self.create_client(ParamSetV2, "/mavros/param/set")
        self.pget_cli = self.create_client(GetParameters, "/mavros/param/get_parameters")
        self.rate_cli = self.create_client(StreamRate, "/mavros/set_stream_rate")

    def on_state(self, m):
        self.state = m

    def on_imu(self, m):
        self.imu = m
        self.imu_t.append(time.monotonic())

    def on_odom(self, m):
        self.odom = m
        self.odom_t.append(time.monotonic())

    def on_set(self, m):
        self.setpoint = m
        self.set_t.append(time.monotonic())

    def on_echo(self, m):
        self.echo = m
        self.echo_t.append(time.monotonic())

    def call(self, client, request, timeout=8.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        end = time.monotonic() + timeout
        while not future.done() and time.monotonic() < end:
            time.sleep(0.02)
        return future.result() if future.done() else None

    def set_mode(self, mode, timeout=6.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            r = SetMode.Request()
            r.custom_mode = mode
            self.call(self.mode_cli, r, 2.0)
            t = time.monotonic() + 1.5
            while time.monotonic() < t:
                if self.state and self.state.mode == mode:
                    return True
                time.sleep(0.02)
        return False

    def set_position_stream(self, hz):
        """MAV_DATA_STREAM_POSITION is 6. Unlike MAV1_POSITION, this applies live."""
        r = StreamRate.Request()
        r.stream_id = 6
        r.message_rate = hz
        r.on_off = True
        return self.call(self.rate_cli, r) is not None

    def get_param(self, name):
        r = GetParameters.Request()
        r.names = [name]
        res = self.call(self.pget_cli, r)
        if res is None or not res.values or res.values[0].type == ParameterType.PARAMETER_NOT_SET:
            return None
        v = res.values[0]
        return v.integer_value if v.type == ParameterType.PARAMETER_INTEGER else v.double_value

    def set_param(self, name, value):
        r = ParamSetV2.Request()
        r.force_set = True
        r.param_id = name
        r.value = (
            ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(value))
            if isinstance(value, int)
            else ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value))
        )
        res = self.call(self.pset_cli, r)
        return bool(res and res.success)

    # Flight data in NED.
    @property
    def pos(self):
        p = self.odom.pose.pose.position
        return T @ np.array([p.x, p.y, p.z])

    @property
    def alt(self):
        return self.odom.pose.pose.position.z

    @property
    def vel(self):
        """World NED velocity, rotating the body-frame twist with the odom attitude."""
        o = self.odom.pose.pose.orientation
        t = self.odom.twist.twist.linear
        return T @ (q2R([o.w, o.x, o.y, o.z]) @ np.array([t.x, t.y, t.z]))

    @property
    def rpy(self):
        return euler_ned(self.imu.orientation)


class Stack:
    def __init__(self):
        self.sitl = self.mavros = self.node = None
        self.node_log = None
        self.rc_on = True
        self.throttle = 1000

    def start(self):
        os.makedirs("/tmp/sitl", exist_ok=True)
        self.sitl = subprocess.Popen(
            [
                "/opt/sitl/arducopter",
                "--model",
                "x",
                "--speedup",
                "1",
                "-w",
                "-I0",
                "--defaults",
                f"/opt/sitl/copter.parm,{TEST}/overrides.parm",
                "--home",
                f"47.397742,8.545594,488,{HEADING_DEG}",
            ],
            cwd="/tmp/sitl",
            stdout=open(OUT / "sitl.log", "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        time.sleep(3)
        self.mavros = subprocess.Popen(
            [
                "ros2",
                "launch",
                "mavros",
                "apm.launch",
                "fcu_url:=tcp://127.0.0.1:5760",
                "tgt_system:=1",
            ],
            stdout=open(OUT / "mavros.log", "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        threading.Thread(target=self.rc_loop, daemon=True).start()

    def rc_loop(self):
        """Sticks centred, through the SITL RC port. Throttle mid means hold height."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while self.rc_on:
            s.sendto(
                struct.pack("<8H", 1500, 1500, self.throttle, 1500, 1000, 1000, 1000, 1000),
                ("127.0.0.1", 5501),
            )
            time.sleep(0.02)

    def start_node(self, model, tag, **params):
        self.stop_node()
        path = str(model) if str(model).startswith("/") else f"{POLICY}/{model}"
        args = [
            "python3",
            "/ws/ros2_inference_node.py",
            "--ros-args",
            "-p",
            f"model_path:={path}",
            "-p",
            f"policy_meta_path:={POLICY}/policy_meta.npz",
        ]
        for k, v in params.items():
            args += ["-p", f"{k}:={v}"]
        self.node_log = OUT / f"node_{tag}.log"
        self.node = subprocess.Popen(
            args, stdout=open(self.node_log, "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid
        )

    def node_text(self):
        return self.node_log.read_text() if self.node_log and self.node_log.exists() else ""

    def wait_log(self, text, timeout=25.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if text in self.node_text():
                return True
            time.sleep(0.1)
        return False

    def stop_node(self, sig=signal.SIGINT):
        if self.node and self.node.poll() is None:
            os.killpg(self.node.pid, sig)
            try:
                self.node.wait(5)
            except subprocess.TimeoutExpired:
                os.killpg(self.node.pid, signal.SIGKILL)
        self.node = None

    def shutdown(self):
        self.rc_on = False
        self.stop_node()
        for p in (self.mavros, self.sitl):
            if p and p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)


def wait_until(fn, timeout, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if fn():
            return True
        time.sleep(step)
    return False


def rate(stamps, window):
    now = time.monotonic()
    recent = [t for t in stamps if t > now - window]
    return len(recent) / window


def settle(p, seconds=7.0):
    """Hold in LOITER and fly back to the test altitude with the throttle stick."""
    p.set_mode("LOITER")
    end = time.monotonic() + 40
    while abs(p.alt - ALT) > 1.0 and time.monotonic() < end:
        STACK[0].throttle = 1250 if p.alt > ALT else 1750
        time.sleep(0.1)
    STACK[0].throttle = 1500
    time.sleep(seconds)


def run_policy(p, s, model, tag, seconds, **params):
    """Hand a constant policy control from LOITER; return samples taken while active."""
    settle(p, 5.0)
    hover = dict(hover_thrust=HOVER[0], max_position_m=500.0)
    hover.update(params)
    s.start_node(model, tag, **hover)
    assert s.wait_log("GUID_OPTIONS verified"), s.node_text()[-500:]
    before = len(p.set_t)
    start = time.monotonic()
    assert p.set_mode("GUIDED")
    assert wait_until(lambda: len(p.set_t) > before, 3.0), "no setpoints: " + s.node_text()[-400:]
    t0 = time.monotonic()
    samples = []
    while time.monotonic() - t0 < seconds:
        samples.append((time.monotonic() - t0, p.rpy, p.vel.copy(), p.alt, p.pos.copy()))
        time.sleep(0.02)
    return samples


HOVER = [0.5]
STACK = [None]


def main():
    POLICY.mkdir(parents=True, exist_ok=True)
    make_const.build(POLICY)
    for name in ("model.onnx", "policy_meta.npz"):
        shutil.copy(USER_POLICY / name, POLICY / name)
    rclpy.init()
    p = Pilot()
    ex = SingleThreadedExecutor()
    ex.add_node(p)
    threading.Thread(target=ex.spin, daemon=True).start()
    s = Stack()
    STACK[0] = s
    s.start()
    try:
        fly(p, s)
    except Exception as e:
        check("suite ran to completion", False, repr(e))
        import traceback

        traceback.print_exc()
    finally:
        s.shutdown()
    print("\n=== RESULTS ===")
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    passed = sum(ok for _, ok, _ in RESULTS)
    print(f"{passed}/{len(RESULTS)} passed", flush=True)
    os._exit(0 if RESULTS and passed == len(RESULTS) else 1)


def fly(p, s):
    head = math.radians(HEADING_DEG)
    forward = np.array([math.cos(head), math.sin(head)])
    right = np.array([-math.sin(head), math.cos(head)])

    # ---- Bring-up
    assert wait_until(lambda: p.state and p.state.connected, 60), "MAVROS never connected"
    log(f"MAVROS connected, mode {p.state.mode}")
    assert wait_until(
        lambda: p.get_param("GUID_TIMEOUT") is not None, 120, 2.0
    ), "params never pulled"
    log(
        f"params pulled. GUID_OPTIONS={p.get_param('GUID_OPTIONS')} GUID_TIMEOUT={p.get_param('GUID_TIMEOUT')} MAV1_EXTRA1={p.get_param('MAV1_EXTRA1')} MAV1_POSITION={p.get_param('MAV1_POSITION')} thrust_scaling config via apm.launch"
    )
    assert wait_until(lambda: p.imu is not None and p.odom is not None, 90), "no imu/odom"
    time.sleep(3)
    check(
        "sensor streams reach 50 Hz over best-effort QoS",
        rate(p.imu_t, 3) > 40 and rate(p.odom_t, 3) > 40,
        f"imu {rate(p.imu_t,3):.1f} Hz, odom {rate(p.odom_t,3):.1f} Hz",
    )

    def arm_and_takeoff():
        for _ in range(45):
            p.set_mode("GUIDED", 3.0)
            r = CommandBool.Request()
            r.value = True
            p.call(p.arm_cli, r, 4.0)
            if wait_until(lambda: p.state.armed, 2.0):
                break
            time.sleep(2)
        assert p.state.armed, "could not arm"
        r = CommandTOL.Request()
        r.altitude = ALT
        res = p.call(p.tol_cli, r)
        log(f"takeoff accepted: {res}")
        assert wait_until(lambda: p.alt > ALT - 1.0, 45), f"never climbed, alt {p.alt}"
        s.throttle = 1500

    arm_and_takeoff()
    settle(p, 6.0)
    yaw = math.degrees(p.rpy[2])
    log(f"airborne: alt {p.alt:.1f} m heading {yaw:.1f} deg mode {p.state.mode}")
    check(
        "spawned at the off-axis test heading", abs(yaw - HEADING_DEG) < 5, f"heading {yaw:.1f} deg"
    )
    n = len(p.throttle)
    time.sleep(4)
    measured = float(np.median(p.throttle[n:]))
    log(
        f"MOT_THST_HOVER param {p.get_param('MOT_THST_HOVER'):.3f}, measured hover throttle {measured:.3f}"
    )
    HOVER[0] = round(measured, 3)

    # ---- GUID_OPTIONS refusal: starts at 0 on purpose
    s.start_node("const_hover.onnx", "guid_refuse", hover_thrust=HOVER[0], max_position_m=500.0)
    check("wrong GUID_OPTIONS is detected", s.wait_log("lacks bit 3"), "")
    before = len(p.set_t)
    p.set_mode("GUIDED")
    time.sleep(3)
    check(
        "wrong GUID_OPTIONS: node publishes nothing in GUIDED",
        len(p.set_t) == before,
        f"{len(p.set_t)-before} messages",
    )
    p.set_mode("LOITER")
    check(
        "GUID_OPTIONS set through MAVROS",
        p.set_param("GUID_OPTIONS", 8),
        str(p.get_param("GUID_OPTIONS")),
    )
    check("node verifies the corrected GUID_OPTIONS", s.wait_log("GUID_OPTIONS verified"), "")
    s.stop_node()

    # ---- Hover and loop rate
    samples = run_policy(p, s, "const_hover.onnx", "hover", 12.0)
    period = np.diff([t for t in p.set_t if t > time.monotonic() - 10]) * 1e3
    log(
        f"period ms: mean {period.mean():.2f} std {period.std():.2f} p99 {np.percentile(period,99):.2f} max {period.max():.2f}"
    )
    late = [l for l in s.node_text().splitlines() if "late ticks" in l]
    check(
        "node reports no late ticks while flying",
        len(late) <= 1,
        late[-1].split("Control loop late: ")[-1] if late else "none",
    )
    check(
        "publishes at 50 Hz",
        48 < 1000 / period.mean() < 52 and np.percentile(period, 99) < 30,
        f"{1000/period.mean():.2f} Hz, std {period.std():.2f} ms, p99 {np.percentile(period,99):.1f} ms, max {period.max():.1f} ms",
    )
    alts = np.array([x[3] for x in samples])
    tilts = np.array([math.hypot(x[1][0], x[1][1]) for x in samples])
    drift = (alts[-1] - alts[0]) / samples[-1][0]
    check(
        "neutral action holds height",
        abs(drift) < 0.5,
        f"{drift:+.2f} m/s over 12 s, hover_thrust {HOVER[0]}",
    )
    check("neutral action stays level", tilts.max() < 0.08, f"max tilt {tilts.max():.3f} rad")
    check(
        "policy stays in control (ArduPilot did not fall back)",
        p.state.mode == "GUIDED" and "not in control" not in s.node_text(),
        p.state.mode,
    )
    yaw_err = max(abs(math.degrees(x[1][2]) - HEADING_DEG) for x in samples)
    check("neutral action holds heading", yaw_err < 8, f"worst error {yaw_err:.1f} deg")
    if p.echo is not None and rate(p.echo_t, 3) > 0:
        check(
            "ArduPilot echoes our attitude target",
            True,
            f"thrust echo {p.echo.thrust:.3f}, sent {p.setpoint.thrust:.3f}, {rate(p.echo_t,3):.0f} Hz",
        )
    else:
        log("ATTITUDE_TARGET echo not streamed")
    s.stop_node()

    # ---- Each axis, in the vehicle's own frame at a 60 degree heading
    samples = run_policy(p, s, "const_roll.onnx", "roll", 2.5)
    roll = np.median([x[1][0] for x in samples[-40:]])
    v = samples[-1][2][:2]
    check(
        "roll: reaches commanded angle", abs(roll - 0.21) < 0.04, f"{roll:.3f} rad, commanded 0.210"
    )
    check(
        "roll: right roll moves the drone to its right",
        v @ right > 1.0 and abs(v @ forward) < 0.5 * (v @ right),
        f"right {v@right:+.2f} forward {v@forward:+.2f} m/s",
    )
    s.stop_node()

    samples = run_policy(p, s, "const_pitch.onnx", "pitch", 2.5)
    pitch = np.median([x[1][1] for x in samples[-40:]])
    v = samples[-1][2][:2]
    check(
        "pitch: reaches commanded angle",
        abs(pitch + 0.21) < 0.04,
        f"{pitch:.3f} rad, commanded -0.210",
    )
    check(
        "pitch: nose down moves the drone forward",
        v @ forward > 1.0 and abs(v @ right) < 0.5 * (v @ forward),
        f"forward {v@forward:+.2f} right {v@right:+.2f} m/s",
    )
    s.stop_node()

    samples = run_policy(p, s, "const_yaw.onnx", "yaw", 3.0)
    yaws = np.unwrap([x[1][2] for x in samples])
    ts = np.array([x[0] for x in samples])
    half = len(ts) // 2
    yaw_rate = np.polyfit(ts[half:], yaws[half:], 1)[0]
    check(
        "yaw: positive rate turns clockwise at 1 rad/s",
        0.7 < yaw_rate < 1.2,
        f"{yaw_rate:+.2f} rad/s, commanded +1.00",
    )
    s.stop_node()

    samples = run_policy(p, s, "const_climb.onnx", "climb", 2.5)
    climb = samples[-1][3] - samples[0][3]
    check("thrust: positive action climbs", climb > 1.0, f"{climb:+.2f} m")
    s.stop_node()

    samples = run_policy(p, s, "const_sink.onnx", "sink", 1.5, min_altitude_m=5.0)
    sink = samples[-1][3] - samples[0][3]
    check("thrust: negative action sinks", sink < -0.5, f"{sink:+.2f} m")
    s.stop_node()

    # ---- Pilot takeover
    run_policy(p, s, "const_hover.onnx", "takeover", 2.0)
    t0 = time.monotonic()
    ok = p.set_mode("LOITER")
    check("pilot takeover changes mode", ok, "")
    silent = wait_until(lambda: time.monotonic() - p.set_t[-1] > 0.3, 3.0)
    check(
        "pilot takeover: setpoints stop",
        silent,
        f"last setpoint {p.set_t[-1]-t0:+.2f} s after the switch was requested",
    )
    # ---- Re-engage after takeover
    before = len(p.set_t)
    p.set_mode("GUIDED")
    time.sleep(1.5)
    check(
        "pilot can hand back after takeover",
        len(p.set_t) > before + 20,
        f"{len(p.set_t)-before} messages",
    )
    s.stop_node()

    # ---- Faults hand over to LOITER
    def handover(tag, model, trigger=None, wait=8.0, **params):
        run_policy(p, s, model, tag, 0.3, **params)
        if trigger:
            trigger()
        t0 = time.monotonic()
        changed = wait_until(lambda: p.state.mode == "LOITER", wait)
        elapsed = time.monotonic() - t0
        reason = [l for l in s.node_text().splitlines() if "not in control" in l]
        check(
            f"fault handover: {tag}",
            changed and reason,
            f"LOITER after {elapsed:.2f} s: {reason[0].split('Policy not in control: ')[-1][:70] if reason else 'NO REASON LOGGED'}",
        )
        n = len(p.set_t)
        time.sleep(1.0)
        check(
            f"fault handover: {tag} stays silent", len(p.set_t) == n, f"{len(p.set_t)-n} messages"
        )
        s.stop_node()

    handover("tilt limit", "const_roll.onnx", max_tilt_rad=0.12)
    settle(p, 6.0)
    edge = float(np.abs(p.pos).max()) + 3.0
    handover("position limit", "const_pitch.onnx", max_position_m=round(edge, 1), wait=10.0)
    settle(p, 6.0)
    floor = round(p.alt - 1.0, 1)
    handover("altitude floor", "const_sink.onnx", min_altitude_m=floor)
    settle(p, 8.0)
    handover("speed limit", "const_pitch.onnx", max_speed_mps=1.5, wait=10.0)

    # Position telemetry drops from 50 Hz to 2 Hz in flight.
    settle(p, 6.0)
    handover(
        "sensor stream slows",
        "const_hover.onnx",
        trigger=lambda: p.set_position_stream(2),
        wait=6.0,
    )
    slowed = rate(p.odom_t, 1.0)
    check(
        "sensor stream fault was really injected",
        slowed < 10,
        f"odom at {slowed:.1f} Hz during the fault",
    )
    p.set_position_stream(50)
    time.sleep(3)
    check("sensor stream restored", rate(p.odom_t, 2) > 40, f"odom {rate(p.odom_t, 2):.1f} Hz")

    # ---- Node killed mid-manoeuvre: ArduPilot must level out by itself
    run_policy(p, s, "const_roll.onnx", "killed", 1.2)
    tilted = abs(p.rpy[0])
    t0 = time.monotonic()
    s.stop_node(signal.SIGKILL)
    levelled = wait_until(lambda: abs(p.rpy[0]) < 0.05, 5.0)
    t_level = time.monotonic() - t0
    check(
        "node killed: ArduPilot levels out after GUID_TIMEOUT",
        tilted > 0.15 and levelled,
        f"roll {tilted:.2f} rad -> level in {t_level:.2f} s, GUID_TIMEOUT 0.5 s",
    )
    time.sleep(1.5)
    a0 = p.alt
    time.sleep(2.0)
    check(
        "node killed: height holds afterwards",
        abs(p.alt - a0) < 1.0,
        f"{p.alt-a0:+.2f} m in 2 s, mode {p.state.mode}",
    )

    # ---- Node restarted while the vehicle sits in GUIDED
    assert p.state.mode == "GUIDED", p.state.mode
    s.start_node("const_pitch.onnx", "restart", hover_thrust=HOVER[0], max_position_m=500.0)
    s.wait_log("GUID_OPTIONS verified")
    before = len(p.set_t)
    time.sleep(3)
    check(
        "restart in GUIDED: does not seize control",
        len(p.set_t) == before and "Policy engaged" not in s.node_text(),
        f"{len(p.set_t)-before} messages",
    )
    s.stop_node()

    # ---- MAVROS freezes: link stops, node stops, ArduPilot times out
    samples = run_policy(p, s, "const_hover.onnx", "mavros_freeze", 1.0)
    os.killpg(s.mavros.pid, signal.SIGSTOP)
    time.sleep(1.0)
    text_end = wait_until(lambda: "not in control" in s.node_text(), 4.0)
    os.killpg(s.mavros.pid, signal.SIGCONT)
    check(
        "MAVROS frozen: node detects it",
        text_end,
        (
            [l for l in s.node_text().splitlines() if "not in control" in l][-1].split("control: ")[
                -1
            ][:60]
            if text_end
            else ""
        ),
    )
    wait_until(lambda: p.state.mode == "LOITER", 5.0)
    log(f"after thaw mode {p.state.mode}")
    s.stop_node()

    # ---- The real exported network
    samples = run_policy(
        p,
        s,
        "model.onnx",
        "trained",
        POLICY_SECONDS,
        max_position_m=200.0,
        max_tilt_rad=0.6,
        max_speed_mps=12.0,
        min_altitude_m=4.0,
    )
    text = s.node_text()
    fault = [l for l in text.splitlines() if "not in control" in l]
    moved = np.linalg.norm(samples[-1][4] - samples[0][4])
    tilts = [math.hypot(x[1][0], x[1][1]) for x in samples]
    check(
        "your policy runs on real telemetry",
        "Policy engaged" in text and "Traceback" not in text,
        f"moved {moved:.1f} m, max tilt {max(tilts):.2f} rad",
    )
    check(
        f"your policy stays in the envelope for {POLICY_SECONDS:.0f} s",
        not fault,
        fault[0].split("control: ")[-1][:70] if fault else "no faults",
    )
    s.stop_node()

    # ---- Land
    p.set_mode("LAND")
    landed = wait_until(lambda: not p.state.armed, 90)
    check("vehicle landed and disarmed", landed, f"alt {p.alt:.2f} m")


if __name__ == "__main__":
    main()
