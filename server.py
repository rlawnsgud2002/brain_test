#!/usr/bin/env python3
"""
EEG Brain Monitor WebSocket Server
Supports: Muse S (via muselsl), TGAM/NeuroSky (via serial), simulation mode
Usage: python server.py [--port 8765] [--device muse|tgam|sim]
"""
import asyncio, json, math, random, time, argparse, logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

try:
    import websockets
except ImportError:
    print("Install: pip install websockets")
    raise

# ── Configuration ──────────────────────────────────────────
CFG = {
    'thLow': 30, 'thHigh': 70, 'dt': 3.0, 'slope': 0.3,
    'device': 'sim', 'port': 8765,
}

# ── V-Pattern State ────────────────────────────────────────
class VPatternDetector:
    def __init__(self):
        self.v_low = False
        self.v_low_time = 0
        self.v_cooldown = 0
        self.history = deque(maxlen=120)

    def step(self, avg, th_low, th_high, dt, slope_thresh=0.3):
        triggered = False
        if self.v_cooldown > 0:
            self.v_cooldown -= 1
        else:
            if not self.v_low and avg < th_low:
                self.v_low = True
                self.v_low_time = time.time()
            if self.v_low and (time.time() - self.v_low_time) > dt:
                self.v_low = False
            if self.v_low and avg > th_high:
                elapsed = time.time() - self.v_low_time
                hist = list(self.history)
                slope = (hist[-1] - hist[-4]) / 3 if len(hist) >= 4 else 0
                if elapsed < dt and slope > slope_thresh:
                    triggered = True
                    self.v_cooldown = int(dt * 10)
                self.v_low = False
        self.history.append(avg)
        return triggered

# ── Simulation Source ──────────────────────────────────────
class SimSource:
    """Generates realistic EEG simulation data (same algorithm as JS frontend)"""
    def __init__(self, n_channels=14):
        self.nch = n_channels
        self.vals = [50.0] * n_channels
        self.t = 0

    def step(self):
        self.t += 0.1
        alpha_osc = math.sin(self.t / 3.2) * 3
        common = (random.random() - 0.5) * 2.8
        for i in range(self.nch):
            noise = (random.random() - 0.5) * 3.2
            self.vals[i] = max(5, min(95, self.vals[i] + common * 0.55 + noise * 0.45))
        avg = sum(self.vals) / self.nch
        focus = avg / 100
        bands = {
            'delta': max(2, min(100, 30 - focus*16 + (random.random()-.5)*8)),
            'theta': max(2, min(100, 38 - focus*18 + alpha_osc + (random.random()-.5)*9)),
            'alpha': max(2, min(100, 58 - focus*42 + alpha_osc + (random.random()-.5)*7)),
            'beta':  max(2, min(100, 24 + focus*52 + (random.random()-.5)*8)),
            'gamma': max(1, min(100, 14 + focus*34 + (random.random()-.5)*7)),
        }
        return self.vals[:], bands

# ── Muse Source (stub — requires muselsl) ─────────────────
class MuseSource:
    """
    Real Muse S connection via muselsl + pylsl.
    Install: pip install muselsl pylsl
    Pair Muse in Bluetooth settings, then run: muselsl stream
    """
    def __init__(self):
        self.inlet = None
        self._fallback = SimSource()
        self._try_connect()

    def _try_connect(self):
        try:
            from pylsl import StreamInlet, resolve_stream
            log.info("Looking for Muse LSL stream...")
            streams = resolve_stream('type', 'EEG', timeout=5)
            if streams:
                self.inlet = StreamInlet(streams[0])
                log.info(f"Connected to Muse: {streams[0].name()}")
            else:
                log.warning("No Muse LSL stream found — falling back to simulation")
        except ImportError:
            log.warning("pylsl not installed — pip install pylsl muselsl")

    def step(self):
        if self.inlet:
            sample, _ = self.inlet.pull_sample(timeout=0.0)
            if sample:
                # TP9=0, AF7=1, AF8=2, TP10=3 — pad to exactly 4 if stream is short
                raw = list(sample[:4]) + [0] * 4
                vals = [max(0, min(100, (v + 500) / 10)) for v in raw[:4]]
                avg = sum(vals) / 4
                bands = {'delta':40,'theta':35,'alpha':50,'beta':35,'gamma':15}
                return vals + [50]*10, bands  # pad to 14ch
        # Fallback simulation — reuse stateful instance for continuity
        return self._fallback.step()

# ── TGAM/NeuroSky Source (stub — requires serial) ─────────
class TGAMSource:
    """
    NeuroSky TGAM via serial port.
    Install: pip install pyserial
    Connect headset, find port: ls /dev/tty.* or Device Manager
    """
    def __init__(self, port='/dev/ttyUSB0', baud=57600):
        self.ser = None
        self._attn = 55
        self._med = 45
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=1)
            log.info(f"TGAM connected on {port}")
        except Exception as e:
            log.warning(f"TGAM serial failed ({e}) — simulation mode")

    def step(self):
        if self.ser:
            # Parse ThinkGear packet (simplified)
            try:
                data = self.ser.read(32)
                # TODO: proper ThinkGear packet parsing
                # For now return simulated values
                pass
            except Exception:
                pass
        # TGAM simulation: slow attention drift with bursts
        if random.random() < 0.015:
            self._attn = min(98, self._attn + random.uniform(5, 15))
        else:
            self._attn += (55 - self._attn) * 0.04 + (random.random() - 0.5) * 4
        self._attn = max(20, min(98, self._attn))
        self._med = max(10, min(95, self._med + (70 - self._attn*0.65 - self._med)*0.06 + (random.random()-0.5)*3))
        bands = {
            'delta':35,'theta':40,'alpha':50,'beta':35,'gamma':10,
            'concentration': self._attn, 'meditation': self._med,
            'poor_signal': 0,
        }
        return [self._attn] + [50]*13, bands

# ── WebSocket Handler ──────────────────────────────────────
async def handle_client(websocket):
    device = CFG['device']
    source = SimSource() if device == 'sim' else (MuseSource() if device == 'muse' else TGAMSource())
    detector = VPatternDetector()
    session_start = time.time()
    connected_clients.add(websocket)
    log.info(f"Client connected | device={device}")

    try:
        async def send_loop():
            try:
                while True:
                    vals, bands = source.step()
                    avg = sum(vals[:4 if device=='muse' else 14]) / (4 if device=='muse' else 14)
                    v_active = detector.step(avg, CFG['thLow'], CFG['thHigh'], CFG['dt'])

                    msg = {
                        'type': 'eeg_data',
                        'channels': vals,
                        'bands': bands,
                        'vpattern': {'v_active': v_active, 'v_low': detector.v_low},
                        'session_time': round(time.time() - session_start, 2),
                    }
                    await websocket.send(json.dumps(msg))
                    await asyncio.sleep(0.1)  # 10Hz update
            except websockets.exceptions.ConnectionClosed:
                pass

        async def recv_loop():
            async for msg in websocket:
                try:
                    d = json.loads(msg)
                    if not isinstance(d, dict):
                        continue
                    if d.get('type') == 'settings':
                        s = d.get('settings', {}) or {}
                        # Validate and coerce numeric fields; reject non-numeric without mutating CFG
                        coerced = {}
                        for k in ('thLow', 'thHigh', 'dt'):
                            if k in s:
                                try: coerced[k] = float(s[k])
                                except (TypeError, ValueError):
                                    log.warning(f"Settings rejected: {k}={s[k]!r} not numeric")
                                    coerced = None; break
                        if coerced is None: continue
                        tl = coerced.get('thLow',  CFG['thLow'])
                        th = coerced.get('thHigh', CFG['thHigh'])
                        if tl >= th:
                            log.warning(f"Settings rejected: thLow ({tl}) >= thHigh ({th})")
                            continue
                        CFG.update(coerced)
                        log.info(f"Settings updated: {CFG}")
                    elif d.get('type') == 'source_change':
                        log.info(f"Source change requested: {d.get('source')}")
                except Exception as e:
                    log.warning(f"Recv error: {e}")

        await asyncio.gather(send_loop(), recv_loop())

    except websockets.exceptions.ConnectionClosed:
        log.info("Client disconnected")
    finally:
        connected_clients.discard(websocket)

connected_clients = set()

async def main():
    parser = argparse.ArgumentParser(description='EEG Brain Monitor WS Server')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', choices=['sim','muse','tgam'], default='sim')
    args = parser.parse_args()
    CFG['device'] = args.device
    CFG['port'] = args.port

    log.info(f"Starting server on ws://localhost:{args.port} | device={args.device}")
    log.info("Open mockup_3d.html and select a LIVE source to connect")
    async with websockets.serve(handle_client, 'localhost', args.port):
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    asyncio.run(main())
