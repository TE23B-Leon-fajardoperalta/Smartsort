# SmartSort - Waste detector (runs on your PC, NOT on the Pico)
# Watches the camera stream, decides "paper / plastic / electronics / nothing"
# and tells the Pico over the USB cable, which then opens the matching bin's servo.
#
# Everything for SmartSort is in one folder, C:\Users\leon.fajardoperalta\SmartSort:
#   detector.py   - this file: run it in Thonny
#   pico_main.py  - the Pico's code (it is on the Pico as main.py and starts by itself)
#   models\       - the pre-trained CLIP model (OpenAI's clip-vit-base-patch32, as ONNX files)
#   old\          - old versions and backups (the old teaching pictures are in old\samples_removed_2026-11-10)
#
# How to run (all in Thonny):
#   1. Plug the Pico into this PC with its USB cable.
#   2. Connect the PC to the AiPi_Cam_D200 Wi-Fi (for the camera picture).
#   3. In Thonny's bottom-right corner choose "Local Python 3" (NOT the Pico):
#      only one program at a time can use the Pico's USB port.
#   4. Press Run (F5). Keep the camera's view EMPTY for about 2 seconds at the start.
#   (It needs opencv-python, numpy, onnxruntime and pyserial - Thonny's Python already has them.)
#
# DASHBOARD: while it runs, http://127.0.0.1:8642 opens in your browser (it works without internet,
# also on the camera's Wi-Fi). It shows the live picture, the AI's answer, what the AI looks at,
# how sure it is of each kind and the log, and has buttons that do the same as the keys below.
# (Use 127.0.0.1, not "localhost": on this PC another program answers on localhost.)
# The camera's picture is shared there too (the camera itself only allows ONE viewer):
#   http://127.0.0.1:8642/api/camera.mjpg   the live camera picture (any number of pages can watch)
#   http://127.0.0.1:8642/api/camera.jpg    the newest camera picture
#   http://127.0.0.1:8642/api/status        everything the detector knows right now (JSON)
#
# To change the Pico's code: Thonny bottom-right "MicroPython (Raspberry Pi Pico)", open
# pico_main.py, File > Save copy... > Raspberry Pi Pico > main.py, then unplug and replug the
# Pico and switch Thonny back to "Local Python 3".
#
# Two states:
#   RED BOX - while it sees nothing, a red box is shown in the middle and the AI only looks in there
#             (people and chairs elsewhere are ignored). Show it something in the red box.
#   FOLLOW  - as soon as it sees paper, plastic or electronics there, the red box disappears and the
#             AI follows what you are holding, wherever you move it. After IDLE_AFTER_S (7) seconds
#             of seeing nothing, it goes back to the red box.
# NO TEACHING NEEDED: the AI is CLIP, a model that already knows what things look like from 400 million
# photos with captions. PROMPTS below is a fixed list of words for every kind ("a plastic bottle",
# "a mobile phone", ...); the AI picks the kind whose words fit the picture best.
# Keys (click the video window first, then press):
#   E = learn the empty view again (if the camera was bumped or moved)
#   L = big label: the LIVE answer for every picture (default), or the delayed answer
#       the bins use (it must hold for about 2 seconds, with the AI at least 50% sure, before a bin opens)
#   B = brighter picture on screen: off / normal / strong (only what you see)
#   F = full screen on/off (you can also drag the window bigger)
#   D = debug view: the zoomed crop the AI looks at
#   S = save a snapshot picture (in the "snapshots" folder) to help tune things
#   Q = quit
# - At EVERY start it first learns the empty view: keep the view EMPTY and the camera STILL for
#   about 2 seconds. If something moved or the picture is blurry, it says so and tries again.
#   The AI then only looks at what is new, so moving the camera or other light doesn't break it.
#   (This is not teaching: it only tells the AI where the background is.)
# - Something it often gets wrong? Add words for it to PROMPTS (in English, like the others).

import collections
import hashlib
import http.client
import json
import math
import os
import re
import subprocess
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import serial
import serial.tools.list_ports

try:
    import onnxruntime   # runs the CLIP model
except ImportError:
    onnxruntime = None

# ---------------- Settings ----------------
CAMERA_STREAM = "http://192.168.169.1/stream"
PICO_PORT = None       # None = find the Pico's USB port by itself, or set it, e.g. "COM3"

EMPTY_EXAMPLES = 10    # pictures used to learn the empty view at the start
CONFIRM_SECONDS = 2.2  # the answer must hold for about 2 seconds before a bin opens (stops flashes)
CONFIRM_SHARE = 0.75   # ...so 3 of every 4 answers in that time must agree (a few odd ones are fine)
MIN_CONFIDENCE = 0.50  # ...and the AI must be at least this sure, or the bins count it as "nothing".
                       # Tested on 24 held-up objects from my old pictures: 12 of 14 waste items opened
                       # the right lid, none the wrong one, and all 10 "nothing" ones stayed closed.
                       # A lid opens for things that aren't waste? Raise MIN_CONFIDENCE (0.6) or
                       # CONFIRM_SHARE (0.85). Real waste doesn't open? Add words for it to PROMPTS.
REPORT_EVERY_S = 0.3   # how often the Pico is told what we see
START_AFTER = 3        # red box -> follow after this many pictures in a row of paper/plastic/electronics
IDLE_AFTER_S = 7       # follow -> back to the red box after this many seconds of seeing nothing

# Picture (press B in the window to switch brightening, F for full screen)
BRIGHTEN = 1           # 0 = off, 1 = normal, 2 = strong (for dark rooms)
BRIGHTEN_NAMES = ("off", "normal", "strong")
BRIGHT_TARGET = (0, 115, 145)   # average brightness a dark picture is lifted to (0-255)
VIEW_WIDTH = None      # window width in pixels; None = fit the screen by itself

# Dashboard (a web page on this PC; it works without internet)
DASHBOARD_PORT = 8642  # http://127.0.0.1:8642 - None = no dashboard (8000 is taken on this PC)
OPEN_DASHBOARD = True  # open the dashboard in the browser at the start (not if a dashboard tab is already open)
SHOW_WINDOW = True     # also show the OpenCV window (False = only the dashboard)

LABELS = ("paper", "plastic", "electronics", "nothing")
WASTE = ("paper", "plastic", "electronics")
NAMES = {"paper": "PAPER", "plastic": "PLASTIC", "electronics": "ELECTRONICS", "nothing": "nothing"}

# The preset library: what each kind can look like, in words the AI understands (English).
# Every line becomes "a photo of <words>." Tested on 282 of my own camera pictures: 89% right.
PROMPTS = {
    "paper": ["a sheet of paper", "a crumpled piece of paper", "a piece of cardboard", "a paper bag",
              "a newspaper", "an envelope", "a paper cup", "a cardboard box", "a receipt",
              "a paper napkin", "a magazine", "a paper towel"],
    "plastic": ["a plastic bottle", "a plastic bag", "a plastic cup", "a plastic container",
                "a plastic wrapper", "a plastic lid", "a plastic straw", "a plastic food package",
                "a yogurt cup", "a plastic fork", "a candy wrapper", "a crisp packet", "a plastic box"],
    "electronics": ["a mobile phone", "a computer mouse", "a cable", "a battery", "a charger", "headphones",
                    "a remote control", "a circuit board", "a keyboard", "a calculator", "a usb stick",
                    "earbuds", "a power bank", "an electronic device"],
    "nothing": ["an empty room", "a hand", "a person", "a wall", "a table", "a ceiling", "a blurry photo",
                "an empty hand", "a face", "a chair", "a sleeve", "fingers", "a dark room"],
}

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")


# ---------------- Log file (what happened, and when) ----------------
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_log.txt")
_log_lock = threading.Lock()
LOG_LINES = collections.deque(maxlen=40)   # the newest lines, for the dashboard


def log(text):
    # Writes one line to detector_log.txt, so a freeze can be explained afterwards
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), text)
    print(line)
    LOG_LINES.append(line)
    try:
        with _log_lock, open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------- Which Wi-Fi is this PC on? ----------------
class WifiWatch:
    # Asks Windows for the Wi-Fi name now and then (often while the camera is frozen),
    # because the camera only works while the PC is on AiPi_Cam_D200
    def __init__(self):
        self.name = None
        self.frozen = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        last = None
        while True:
            try:
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-NetConnectionProfile | Where-Object InterfaceAlias -eq 'Wi-Fi').Name"],
                    capture_output=True, text=True, timeout=10, creationflags=0x08000000)  # no console window
                self.name = out.stdout.strip() or "no Wi-Fi"
            except Exception:
                self.name = None
            if self.name != last:
                log("Wi-Fi: %s" % self.name)
                last = self.name
            time.sleep(3 if self.frozen else 20)


# ---------------- Camera (reads the MJPEG stream in the background) ----------------
CAMERA_PROBLEMS = {
    "busy": "the camera refuses the picture (busy)",
    "unreachable": "the camera can't be reached",
    "stalled": "the camera stopped sending",
    "broken": "the picture stream broke off",
}


def camera_problem(error, had_pictures):
    # Sorts a camera error into one of CAMERA_PROBLEMS
    cause = getattr(error, "reason", error)        # urllib wraps the real error
    text = str(cause)
    if isinstance(cause, http.client.IncompleteRead):
        return "broken"
    if (isinstance(cause, (ConnectionResetError, ConnectionAbortedError, http.client.RemoteDisconnected))
            or "10054" in text or "10053" in text or "without response" in text):
        # The camera answered and then closed at once: it only sends its picture to ONE
        # viewer, and it thinks someone else (or an old connection) still has it
        return "broken" if had_pictures else "busy"
    if "timed out" in text:
        return "stalled" if had_pictures else "unreachable"
    return "unreachable"


class Camera:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.jpeg = None            # the newest picture exactly as the camera sent it (for the dashboard)
        self.frame_id = 0
        self.last_time = 0          # when the last picture arrived
        self.status = "connecting..."
        self.problem = None         # one of CAMERA_PROBLEMS while there are no pictures
        self.problem_since = None   # when the pictures stopped
        self.stop = False           # True = let go of the camera (it allows only ONE viewer)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        logged, tries, summary_at = None, 0, 0
        while not self.stop:
            had_pictures = False
            detail = ""
            try:
                # timeout: if no data arrives for 3 s, give up on this connection and reconnect
                with urllib.request.urlopen(self.url, timeout=3) as stream:
                    self.status = "connected, waiting for pictures"
                    buf = b""
                    while not self.stop:
                        chunk = stream.read(4096)
                        if not chunk:
                            break
                        buf += chunk
                        # Every JPEG picture starts with FFD8 and ends with FFD9
                        while True:
                            start = buf.find(b"\xff\xd8")
                            if start == -1:
                                buf = buf[-1:]
                                break
                            end = buf.find(b"\xff\xd9", start + 2)
                            if end == -1:
                                buf = buf[start:]
                                break
                            jpg = buf[start:end + 2]
                            buf = buf[end + 2:]
                            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                self.frame = img
                                self.jpeg = jpg
                                self.frame_id += 1
                                self.last_time = time.time()
                                had_pictures = True
                                if self.status != "ok":
                                    if self.problem_since is not None:
                                        log("Camera: pictures arriving again (after %d s, %d tries)"
                                            % (time.time() - self.problem_since, tries))
                                    else:
                                        log("Camera: pictures arriving")
                                    self.status = "ok"
                                    self.problem, self.problem_since = None, None
                                    logged, tries = None, 0
                if self.stop:
                    break
                kind = "stalled" if had_pictures else "busy"   # the camera closed the stream itself
            except Exception as e:
                if self.stop:
                    break
                kind = camera_problem(e, had_pictures)
                detail = str(getattr(e, "reason", e))[:80]
            now = time.time()
            if self.problem_since is None:
                self.problem_since = now
            self.problem = kind
            self.status = CAMERA_PROBLEMS[kind]
            tries += 1
            # The log gets each NEW kind of problem once, then a short summary every 30 s
            if kind != logged:
                log("Camera: %s%s" % (self.status, " (%s)" % detail if detail else ""))
                logged, summary_at = kind, now + 30
            elif now >= summary_at:
                log("Camera: still no pictures for %d s - %s (%d tries)" % (now - self.problem_since, self.status, tries))
                summary_at = now + 30
            time.sleep(2 if kind == "busy" else 0.5)   # a busy camera gets a little more time


BUSY_HELP_1 = "The camera refuses the picture (busy) for %d s"
BUSY_HELP_2 = "Close any browser/phone showing the camera. Still busy after 30 s? Unplug + replug the Pico"


# ---------------- Pico (talks over the USB cable in the background) ----------------
class Pico:
    def __init__(self, port=None):
        self.port = port
        self.item = "nothing"
        self.online = False
        self.error = "looking for the Pico..."
        self.message = ""    # the last thing the Pico printed, e.g. "plastic! Bin open"
        self.stop = False    # True = let go of the USB port (so Thonny can talk to the Pico again)
        threading.Thread(target=self._run, daemon=True).start()

    def find_port(self):
        if self.port:
            return self.port
        for p in serial.tools.list_ports.comports():
            if p.vid == 0x2E8A:   # Raspberry Pi's USB id: this is the Pico
                return p.device
        return None

    def _run(self):
        while not self.stop:
            name = self.find_port()
            if name is None:
                self.online = False
                self.error = "Pico not found - is its USB cable plugged into this PC?"
                time.sleep(2)
                continue
            try:
                self._talk(name)
            except serial.SerialException as e:
                self.online = False
                if "PermissionError" in str(e):   # Windows: another program has the port open
                    self.error = "%s is busy - is Thonny connected to the Pico? Close it." % name
                else:
                    self.error = "%s: %s" % (name, e)
                self._log_state()
            time.sleep(2)   # unplugged or busy: try again

    def _log_state(self):
        # Writes to the log only when the Pico's state changes
        state = "connected" if self.online else "NOT connected (%s)" % self.error
        if state != getattr(self, "_logged", None):
            log("Pico: " + state)
            self._logged = state

    def _talk(self, name):
        with serial.Serial(name, 115200, timeout=0.1, write_timeout=1) as s:
            opened = time.time()
            last_sent, last_time, last_ok, buf = None, 0, 0, b""
            while not self.stop:
                item = self.item
                if item != last_sent or time.time() - last_time > REPORT_EVERY_S:
                    s.write(("item %s\n" % item).encode())
                    last_sent, last_time = item, time.time()
                buf += s.read(256)
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode(errors="replace").strip()
                    if text.startswith("ok"):
                        last_ok = time.time()      # the Pico answered
                    elif text and text != self.message:
                        self.message = text
                        print("Pico:", text)
                self.online = time.time() - last_ok < 3
                if not self.online and time.time() - opened > 3:
                    self.error = "%s: no answer - unplug and replug the Pico so main.py starts" % name
                if self.online or time.time() - opened > 3:
                    self._log_state()


# ---------------- Brighter pictures ----------------
def brighten(img, level=None):
    # Makes a dark picture easier to see: lifts the dark parts (gamma) until the
    # average brightness reaches BRIGHT_TARGET, then boosts local contrast (CLAHE).
    # A picture that is already bright is hardly changed.
    level = BRIGHTEN if level is None else level
    if level == 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean = float(l.mean())
    target = BRIGHT_TARGET[level]
    if mean < target:
        gamma = np.log(target / 255.0) / np.log(max(mean, 1.0) / 255.0)
        gamma = max(gamma, 0.25 if level == 2 else 0.4)   # don't blow up pure noise
        lut = (np.power(np.arange(256) / 255.0, gamma) * 255).astype(np.uint8)
        l = cv2.LUT(l, lut)
    tiles = max(2, min(8, img.shape[1] // 40))
    clahe = cv2.createCLAHE(clipLimit=2.0 if level == 1 else 3.0, tileGridSize=(tiles, tiles))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ---------------- The AI ----------------
# 1. At every start the detector learns the EMPTY view (the camera must be still and the view empty).
# 2. In every picture it finds what is NEW compared with that view (clearly darker or brighter),
#    and focuses on the biggest new thing near the middle (people far away at the edges count less).
# 3. It cuts that part out of the full-size picture with a small margin: a ZOOMED CROP.
# 4. CLIP compares the crop with the words in PROMPTS and says how well each kind fits:
#    paper / plastic / electronics / nothing. Nothing new in the red box is always "nothing".
AI_W, AI_H = 160, 120   # the AI looks for changes on a small copy of the picture
TOL_PX = 8          # the camera may wobble this many pixels without the AI seeing "new" things
T_DARK = 0.25       # "new" = clearly darker or brighter than the empty view (log units, about 22%) ...
T_CHAN = 0.35       # ... or one colour clearly different (red or green labels)
MIDDLE_BOX = (0.25, 0.30, 0.75, 0.75)  # the box the AI looks in: left, top, right, bottom (part of the picture)
EMPTY_SHARE = 0.002 # fewer new pixels than 0.2% of the picture -> nothing new in the middle box
PAD = 0.18          # margin around the object in the crop (part of the object's size)
CROP_MAX = 256      # zoomed crops: longest side in pixels
MOTION_LIMIT = 4.0  # empty view: pictures may differ this much (still camera ~1.3, moving hand ~4.5)
MIN_SHARPNESS = 50  # empty view: below this the picture is blurry or the lens is covered
LAMP_LOG = math.log(215 + 4)   # empty-view pixels this bright (lamps, windows) can't count as "brighter"
SAT_LOG = math.log(245 + 4)    # pure white (a lamp): white in the empty view and now = unchanged

_yy, _xx = np.mgrid[0:AI_H, 0:AI_W]
_EDGE = np.minimum(np.minimum(_yy, AI_H - 1 - _yy), np.minimum(_xx, AI_W - 1 - _xx)).astype(np.float32)
_MX0, _MY0 = int(round(MIDDLE_BOX[0] * AI_W)), int(round(MIDDLE_BOX[1] * AI_H))
_MX1, _MY1 = int(round(MIDDLE_BOX[2] * AI_W)), int(round(MIDDLE_BOX[3] * AI_H))
_ZONE = np.zeros((AI_H, AI_W), np.float32)
_ZONE[_MY0:_MY1, _MX0:_MX1] = 1                    # only things that reach into the middle box count
GROW = 0.4          # a thing in the middle box may stick out of it by this much (part of the box size)
_GX, _GY = int(GROW * (_MX1 - _MX0)), int(GROW * (_MY1 - _MY0))
_LIMIT = np.zeros((AI_H, AI_W), bool)              # ... but never further than this
_LIMIT[max(0, _MY0 - _GY):_MY1 + _GY, max(0, _MX0 - _GX):_MX1 + _GX] = True
_LIM_W, _LIM_H = _MX1 - _MX0 + 2 * _GX, _MY1 - _MY0 + 2 * _GY
TRACK_GROW = 0.5    # follow state: look this far around where the held thing was (part of its size)
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOL_PX + 1, 2 * TOL_PX + 1))
_OPEN3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_JOIN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_bgcache = {}


def small_copy(frame):
    # The picture the AI looks for changes in: 160 x 120, shrunk by averaging (removes grainy noise)
    return cv2.resize(frame, (AI_W, AI_H), interpolation=cv2.INTER_AREA)


def _bgprep(bg):
    # Work that is done only once per empty view
    key = hashlib.md5(bg.tobytes()).digest()
    c = _bgcache.get(key)
    if c is None:
        lb = np.log(cv2.GaussianBlur(bg, (3, 3), 0).astype(np.float32) + 4)
        c = (lb, cv2.erode(lb, _KERNEL), cv2.dilate(lb, _KERNEL))   # darkest / brightest nearby
        if len(_bgcache) > 8:
            _bgcache.clear()
        _bgcache[key] = c
    return c


def change_map(small, bg):
    # 1 for every NEW pixel (clearly darker or brighter than the empty view, after removing
    # the camera's own brightness change)
    lb, lo, hi = _bgprep(bg)
    la = np.log(cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32) + 4)
    gain = np.median((la - lb)[::3, ::3].reshape(-1, 3), axis=0)
    darker = (lo + gain) - la
    brighter = la - (hi + gain)
    # "Brighter" doesn't count next to lamps and windows (nearly white in the empty view): their
    # glow grows and shrinks when the camera changes its brightness
    lit = hi.max(2) > LAMP_LOG
    # A lamp that is pure white in the empty view AND now is the same lamp (it can't follow the
    # camera's brightness change, so it would otherwise look "darker")
    same_lamp = (la.max(2) > SAT_LOG) & (lb.max(2) > SAT_LOG)
    fg = (((darker.mean(2) > T_DARK) | (darker.max(2) > T_CHAN)
           | (((brighter.mean(2) > T_DARK) | (brighter.max(2) > T_CHAN)) & ~lit)) & ~same_lamp)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, _OPEN3)
    return fg.astype(np.float32)


def focus_box(weights, zone=_ZONE, limit=_LIMIT):
    # The new thing that reaches furthest into the zone (the middle box) -> ((x0, y0, x1, y1) on the
    # 160 x 120 copy, its size, its pixels, "object"), or None when nothing new is in the zone.
    # Things that stay outside the zone (people, chairs at the edges) are ignored; a thing in it
    # is taken whole, even where it sticks out (but never past the limit).
    if (weights * zone).sum() / (AI_W * AI_H) < EMPTY_SHARE:
        return None
    m = (weights > 0).astype(np.uint8)
    n, lab, _, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, _JOIN), 8)   # parts of one object join up
    if n <= 1:
        return None
    inside = np.bincount(lab.ravel(), weights=(weights * zone).ravel(), minlength=n)
    best = int(np.argmax(inside[1:])) + 1
    obj = (lab == best) & (m > 0) & limit
    ys, xs = np.nonzero(obj)
    if len(xs) == 0:
        return None
    pad = PAD * max(xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()) + 2   # a margin around it
    x0, y0 = max(0, int(xs.min() - pad)), max(0, int(ys.min() - pad))
    x1, y1 = min(AI_W, int(math.ceil(xs.max() + 1 + pad))), min(AI_H, int(math.ceil(ys.max() + 1 + pad)))
    return (x0, y0, x1, y1), float(obj.sum()) / (AI_W * AI_H), obj, "object"


def _window(cx, cy, w, h):
    m = np.zeros((AI_H, AI_W), bool)
    m[max(0, int(cy - h / 2)):int(cy + h / 2) + 1, max(0, int(cx - w / 2)):int(cx + w / 2) + 1] = True
    return m


def track_area(box):
    # Follow state: where to look for the held thing now -> (zone, limit). The zone is around where
    # it was in the last picture; the limit (how far it may reach) is never bigger than in the red box
    x0, y0, x1, y1 = box
    w, h, cx, cy = x1 - x0, y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2
    zone = _window(cx, cy, min(w * (1 + 2 * TRACK_GROW) + 8, _LIM_W), min(h * (1 + 2 * TRACK_GROW) + 8, _LIM_H))
    limit = _window(cx, cy, min(w * (1 + 4 * TRACK_GROW) + 8, _LIM_W), min(h * (1 + 4 * TRACK_GROW) + 8, _LIM_H))
    return zone.astype(np.float32), limit


def middle_box():
    # The whole middle box: what the AI looks at when nothing new is in it
    obj = np.zeros((AI_H, AI_W), bool)
    obj[_MY0:_MY1, _MX0:_MX1] = True
    return (_MX0, _MY0, _MX1, _MY1), 0.0, obj, "middle"


def make_crop(frame, found):
    # The zoomed crop of the full-size picture; its 4th channel marks the object (the rest is background)
    (x0, y0, x1, y1), area, obj = found[:3]
    sx, sy = frame.shape[1] / AI_W, frame.shape[0] / AI_H
    X0, Y0, X1, Y1 = int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)
    crop = frame[Y0:Y1, X0:X1]
    mask = cv2.dilate(obj[y0:y1, x0:x1].astype(np.uint8), np.ones((3, 3), np.uint8))
    mask = cv2.resize(mask, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_NEAREST)
    s = min(1.0, CROP_MAX / max(X1 - X0, Y1 - Y0))
    if s < 1.0:
        size = (max(1, int((X1 - X0) * s)), max(1, int((Y1 - Y0) * s)))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return np.dstack([crop, mask * 255]).astype(np.uint8)


# ---- the pre-trained model: CLIP (OpenAI's clip-vit-base-patch32) ----
# CLIP learned from 400 million photos with captions. It turns a picture into 512 numbers, and a
# sentence into 512 numbers in the same "language": the closer they are, the better the sentence
# describes the picture. Nothing is learned here - the words in PROMPTS are the whole library.
CLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "clip-vit-base-patch32")
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
CLIP_SCALE = 100.0      # CLIP's own sharpness for turning closeness into percentages
PROMPT_CACHE = os.path.join(CLIP_DIR, "_prompts_cache.npz")   # the words' numbers, so later starts are fast


def _bytes_to_unicode():
    # CLIP's word pieces write every byte as a visible letter
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class ClipWords:
    # CLIP's tokenizer: turns a sentence into the word-piece numbers its text model reads
    # (vocab.json and merges.txt come with the model)
    PATTERN = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\sa-z0-9]+")

    def __init__(self, folder):
        with open(os.path.join(folder, "vocab.json"), encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(os.path.join(folder, "merges.txt"), encoding="utf-8") as f:
            merges = [tuple(m.split()) for m in f.read().split("\n")[1:] if m.strip()]
        self.ranks = dict(zip(merges, range(len(merges))))
        self.letters = _bytes_to_unicode()

    def _pieces(self, word):
        parts = tuple(word[:-1]) + (word[-1] + "</w>",)
        while len(parts) > 1:
            pair = min(zip(parts, parts[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            joined, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == pair:
                    joined.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    joined.append(parts[i])
                    i += 1
            parts = tuple(joined)
        return parts

    def __call__(self, text):
        ids = [self.vocab["<|startoftext|>"]]
        for word in self.PATTERN.findall(" ".join(text.lower().split())):
            word = "".join(self.letters[b] for b in word.encode("utf-8"))
            ids += [self.vocab[p] for p in self._pieces(word)]
        return ids + [self.vocab["<|endoftext|>"]]


def _session(name):
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3                                    # no warnings in Thonny
    options.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)  # one core stays free for the window
    return onnxruntime.InferenceSession(os.path.join(CLIP_DIR, "onnx", name), options,
                                        providers=["CPUExecutionProvider"])


def prompt_numbers():
    # The 512 numbers of every kind in PROMPTS: the average of its sentences (worked out once, then cached)
    sentences = {label: ["a photo of %s." % words for words in PROMPTS[label]] for label in LABELS}
    key = hashlib.md5(json.dumps(sentences, sort_keys=True).encode()).hexdigest()
    try:
        with np.load(PROMPT_CACHE) as z:
            if str(z["key"]) == key:
                return z["numbers"]
    except Exception:
        pass                                  # no cache yet, or PROMPTS changed
    words, text_model = ClipWords(CLIP_DIR), _session("text_model.onnx")
    rows = []
    for label in LABELS:
        e = np.array([text_model.run(None, {"input_ids": np.array([words(s)], np.int64)})[0][0]
                      for s in sentences[label]])
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        m = e.mean(0)
        rows.append(m / np.linalg.norm(m))
    numbers = np.array(rows, np.float32)
    try:
        np.savez(PROMPT_CACHE, key=key, numbers=numbers)
    except OSError:
        pass                                  # only makes the next start slower
    return numbers


def clip_input(crop):
    # The zoomed crop as CLIP wants it: square (padded with its own middle colour, so the object
    # isn't squashed), 224 x 224 pixels, RGB
    bgr = crop[..., :3]
    h, w = bgr.shape[:2]
    side = max(h, w)
    sq = np.empty((side, side, 3), np.uint8)
    sq[:] = np.median(bgr.reshape(-1, 3), axis=0).astype(np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = bgr
    rgb = cv2.cvtColor(cv2.resize(sq, (224, 224), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def check_empty_view(smalls):
    # Is this a good empty view? The camera must be still, nothing moving, and the picture sharp.
    grays = np.stack([cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32) for s in smalls])
    median = np.median(grays, axis=0)
    motion = float(np.percentile([np.abs(g - median).mean() for g in grays], 80))
    sharp = float(cv2.Laplacian(median.astype(np.uint8), cv2.CV_64F).var())
    if motion > MOTION_LIMIT:
        return False, "something moved (%.1f) - keep the camera still and the view empty" % motion
    if sharp < MIN_SHARPNESS:
        return False, "the picture is blurry (%.0f) - is something right in front of the lens?" % sharp
    return True, "still %.1f, sharp %.0f" % (motion, sharp)


ONLY_NOTHING = {label: float(label == "nothing") for label in LABELS}
NO_SCORES = dict.fromkeys(LABELS, 0.0)


class WasteAI:
    # The preset library: CLIP and the words in PROMPTS. Nothing is taught, saved or learned
    # (except the empty view at every start, which only shows the AI where the background is).

    def __init__(self):
        self.bg = None                      # the empty view of THIS run (learned at start-up)
        self.error = None                   # why the AI can't work (then it always says "nothing")
        self.vision = self.words = None
        if onnxruntime is None:
            self.error = "onnxruntime is missing (Thonny: Tools > Manage packages > onnxruntime)"
        elif not os.path.exists(os.path.join(CLIP_DIR, "onnx", "vision_model.onnx")):
            self.error = "the CLIP model is missing: " + CLIP_DIR
        else:
            try:
                self.words = prompt_numbers()
                self.vision = _session("vision_model.onnx")
            except Exception as e:
                self.error = "the CLIP model could not be loaded (%s)" % e

    def set_view(self, bg):
        self.bg = bg                        # the empty view of this run (learned at start-up, or with E)

    def focus(self, small, follow=None):
        # What the AI looks at. Red-box state (follow is None): the new thing in the red box, or else
        # the whole red box. Follow state (follow = where the held thing was): that thing, or else
        # the same as in the red-box state
        if self.bg is None:
            return None
        weights = change_map(small, self.bg)
        if follow is not None:
            return focus_box(weights, *track_area(follow)) or focus_box(weights) or middle_box()
        return focus_box(weights) or middle_box()

    def predict(self, frame, found):
        # Returns (label, confidence 0..1, how well every kind fits). About 0.2 s when there is
        # something new to look at, at once when there isn't
        if self.vision is None or self.bg is None:
            return "nothing", 0.0, dict(NO_SCORES)       # no AI: never open a bin
        if found is None or found[3] == "middle":
            return "nothing", 1.0, dict(ONLY_NOTHING)    # nothing new in the red box: an empty table never opens a bin
        e = self.vision.run(None, {"pixel_values": clip_input(make_crop(frame, found))})[0][0]
        s = self.words @ (e / (np.linalg.norm(e) + 1e-9)) * CLIP_SCALE
        p = np.exp(s - s.max())
        p /= p.sum()
        best = int(np.argmax(p))
        return LABELS[best], float(p[best]), {label: round(float(v), 3) for label, v in zip(LABELS, p)}


class Thinker:
    # CLIP needs about 0.2 s per picture, so it thinks in the background and the window stays smooth.
    # It always takes the NEWEST picture and skips the ones that arrived while it was busy.
    def __init__(self, ai):
        self.ai = ai
        self.job = None              # (picture, what to look at) waiting for the AI
        self.answer = None           # (label, confidence, scores, what it looked at) not picked up yet
        self.look_s = 0.0            # how long the last look took
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def give(self, frame, found):
        with self._cv:
            self.job = (frame, found)
            self._cv.notify()

    def take(self):
        # The newest answer, once (None = no new answer yet)
        with self._cv:
            answer, self.answer = self.answer, None
        return answer

    def _run(self):
        while True:
            with self._cv:
                while self.job is None:
                    self._cv.wait()
                frame, found = self.job
                self.job = None
            started = time.time()
            try:
                label, conf, scores = self.ai.predict(frame, found)
            except Exception as e:                   # the AI must never stop for good
                log("AI error: %s" % e)
                label, conf, scores = "nothing", 0.0, dict(NO_SCORES)
                time.sleep(1)
            self.look_s = time.time() - started
            with self._cv:
                self.answer = (label, conf, scores, found)


# ---------------- Window ----------------
WINDOW = "SmartSort waste detector"


def screen_size():
    # The screen's size in pixels (Windows); a safe guess anywhere else
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1280, 800


def view_width():
    if VIEW_WIDTH:
        return VIEW_WIDTH
    w, h = screen_size()
    # As big as fits: 90% of the screen width, or 85% of its height for a 4:3 picture
    return int(min(w * 0.9, h * 0.85 * 4 / 3))


def put(img, text, y, color, scale=0.6):
    # Writes text on a dark box. Positions and sizes are given for a 640-wide picture
    # and grow with the window, so the text stays readable on a big screen.
    s = img.shape[1] / 640
    thick = max(1, round((2 if scale >= 1 else 1) * s))
    scale *= s
    x, y = int(10 * s), int(y * s)
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = max(2, int(4 * s))
    cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad // 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def show_debug(view, frame, ai, found):
    # D: shows the ZOOMED CROP the AI looks at; the background around the object is darker
    WHITE = (255, 255, 255)
    s = view.shape[1] / 640
    box = int(180 * s)
    top, right = int(155 * s), view.shape[1] - int(10 * s)
    if ai.bg is None:
        put(view, "D: learning the empty view first", 175, WHITE)
        return
    if found is None:
        put(view, "D: nothing new in view - nothing to crop", 175, WHITE)
        return
    crop = make_crop(frame, found)
    shade = 0.35 + 0.65 * (crop[..., 3:4] / 255.0)              # background darker, object bright
    pic = (crop[..., :3] * shade).astype(np.uint8)
    h, w = pic.shape[:2]
    k = box / max(h, w)
    pic = cv2.resize(pic, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_LINEAR)
    view[top:top + pic.shape[0], right - pic.shape[1]:right] = pic
    put(view, "D: the zoomed crop the AI looks at", 175, WHITE)


# ---------------- Dashboard (a web page and a small API, only on this PC) ----------------
DASH_KEYS = "ebsq"             # the keys the dashboard's buttons may press (the same as in the window)


class _DashServer(ThreadingHTTPServer):
    # Windows would otherwise let a second program use the same port, and the wrong one would answer
    allow_reuse_address = False


class Dashboard:
    # A tiny web server in the background: the dashboard page, the camera's picture and the
    # detector's answers. Only this PC can open it (127.0.0.1). The page's buttons come back
    # here as key presses, so they do exactly what the keys in the window do.
    def __init__(self, port):
        self.port = None
        self.closed = False
        self.status = {}              # everything the page shows (made new for every loop)
        self.view_jpg = None          # the picture with the red box / focus box (JPEG)
        self.crop_jpg = None          # the zoomed crop the AI looks at (JPEG)
        self.camera = None            # main() sets this: the raw pictures come from the camera
        self.keys = collections.deque(maxlen=20)
        self.viewers = 0              # open video streams
        self.asked = 0                # when the page last asked for something
        self._lock = threading.Lock()
        self._new = threading.Condition()
        self._view_at = self._crop_at = 0.0
        if not port:
            return
        for p in range(port, port + 5):   # busy (for example another detector still running)? try the next
            try:
                self.server = _DashServer(("127.0.0.1", p), _dashboard_handler(self))
                self.port = p
                break
            except OSError:
                continue
        if self.port is None:
            log("Dashboard: ports %d-%d are all busy - no dashboard this time" % (port, port + 4))
            return
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        log("Dashboard: %s" % self.url())

    def url(self):
        # 127.0.0.1, the address it really listens on ("localhost" can reach another program)
        return "http://127.0.0.1:%d" % self.port

    def open_browser_later(self):
        # Opens the dashboard after 2.5 s - unless a tab that was already open has reconnected by then
        started = time.time()
        threading.Timer(2.5, lambda: self.asked < started and webbrowser.open(self.url())).start()

    def watched(self):
        # Pictures for the page are only made while someone has it open
        return self.port is not None and (self.viewers > 0 or time.time() - self.asked < 3)

    def set_view(self, view):
        # The picture with the boxes, at most 15 times a second and at most 960 pixels wide
        if time.time() - self._view_at < 1 / 15:
            return
        self._view_at = time.time()
        h, w = view.shape[:2]
        if w > 960:
            view = cv2.resize(view, (960, h * 960 // w), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.view_jpg = jpg.tobytes()

    def set_crop(self, frame, found):
        # What the AI looks at (the zoomed crop; the background around the object darker), 4 times a second
        if time.time() - self._crop_at < 0.25:
            return
        self._crop_at = time.time()
        if found is None:
            self.crop_jpg = None
            return
        crop = make_crop(frame, found)
        shade = 0.35 + 0.65 * (crop[..., 3:4] / 255.0)
        ok, jpg = cv2.imencode(".jpg", (crop[..., :3] * shade).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.crop_jpg = jpg.tobytes() if ok else None

    def tick(self):
        # Once per loop: wakes up the video streams
        with self._new:
            self._new.notify_all()

    def wait(self, timeout):
        with self._new:
            self._new.wait(timeout)

    def next_key(self):
        # A button pressed on the dashboard (255 = none, like cv2.waitKey)
        try:
            return self.keys.popleft()
        except IndexError:
            return 255

    def close(self):
        self.closed = True
        if self.port is not None:
            self.server.shutdown()        # stops answering
            self.server.server_close()    # frees the port for the next start


def _dashboard_handler(dash):
    class Handler(BaseHTTPRequestHandler):
        timeout = 20                      # drop connections that stop talking or reading

        def log_message(self, *args):
            pass                          # no line in Thonny for every request

        def end_headers(self):
            # Other web sites may not show the dashboard inside their page (they could steal clicks)
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            super().end_headers()

        def _from_this_pc(self):
            # The address must be localhost: stops other web sites from using it through the browser
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            return host in ("localhost", "127.0.0.1")

        def _reply(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._from_this_pc():
                return self._reply(403, b"Only for this PC", "text/plain")
            path = self.path.split("?", 1)[0]
            dash.asked = time.time()
            try:
                if path == "/":
                    self._reply(200, DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/api/status":
                    self._reply(200, json.dumps(dash.status).encode("utf-8"), "application/json")
                elif path in ("/api/camera.jpg", "/api/view.jpg", "/api/crop.jpg"):
                    jpg = {"/api/camera.jpg": dash.camera.jpeg if dash.camera else None,
                           "/api/view.jpg": dash.view_jpg, "/api/crop.jpg": dash.crop_jpg}[path]
                    if path == "/api/camera.jpg" and jpg and time.time() - dash.camera.last_time > 3:
                        self._reply(503, b"The camera sends no new pictures", "text/plain")
                    elif jpg:
                        self._reply(200, jpg, "image/jpeg")
                    else:
                        self._reply(404, b"No picture yet", "text/plain")
                elif path in ("/api/view.mjpg", "/api/camera.mjpg"):
                    self._stream(path)
                else:
                    self._reply(404, b"Not found", "text/plain")
            except (ConnectionError, OSError):
                pass                      # the page was closed

        def _stream(self, path):
            # MJPEG: one never-ending answer with one JPEG after the other (an <img> shows it as video)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with dash._lock:
                dash.viewers += 1
            last, sent = None, 0.0
            try:
                while not dash.closed:
                    dash.wait(1.0)
                    if path == "/api/view.mjpg":
                        jpg = dash.view_jpg
                    else:
                        jpg = dash.camera.jpeg if dash.camera else None
                    # A new picture - or the old one again every 2 s, so a closed page is noticed
                    if jpg is None or (jpg is last and time.time() - sent < 2):
                        continue
                    last, sent = jpg, time.time()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg + b"\r\n")
            finally:
                with dash._lock:
                    dash.viewers -= 1

        def do_POST(self):
            # A button on the page: {"key": "1"}. Only the page itself can send this (custom header)
            if not self._from_this_pc() or self.headers.get("X-SmartSort") != "1":
                return self._reply(403, b"Only for the dashboard", "text/plain")
            if self.path != "/api/key":
                return self._reply(404, b"Not found", "text/plain")
            try:
                try:
                    n = max(0, min(int(self.headers.get("Content-Length") or 0), 1000))
                    key = str(json.loads(self.rfile.read(n) or b"{}").get("key", ""))
                except (ValueError, AttributeError):
                    key = ""
                if len(key) == 1 and key in DASH_KEYS:
                    if ord(key) not in dash.keys:   # a held button: at most one press waits (like a held key)
                        dash.keys.append(ord(key))
                    self._reply(200, b'{"ok": true}', "application/json")
                else:
                    self._reply(400, b'{"ok": false}', "application/json")
            except (ConnectionError, OSError):
                pass                          # the page was closed

    return Handler


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmartSort</title>
<style>
:root { --bg:#0f1115; --card:#181b22; --line:#2a2f3a; --text:#e8eaf0; --dim:#9aa1b2;
        --paper:#b4e6ff; --plastic:#3ca0ff; --electronics:#3cc83c; --nothing:#f05050; --warn:#ffd23c; --ok:#50dc50; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.4 "Segoe UI", system-ui, sans-serif; }
header { display:flex; align-items:center; gap:10px; padding:10px 16px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
h1 { font-size:18px; margin:0 8px 0 0; }
.chip { padding:3px 10px; border-radius:999px; background:var(--card); border:1px solid var(--line); font-size:13px; color:var(--dim); }
.chip.ok { color:var(--ok); border-color:#2d5a2d; } .chip.bad { color:var(--nothing); border-color:#5a2d2d; }
main { display:grid; grid-template-columns:minmax(0, 1fr) 340px; gap:16px; padding:16px; }
@media (max-width:900px) { main { grid-template-columns:1fr; } }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; }
.video { position:relative; background:#000; border-radius:8px; overflow:hidden; aspect-ratio:4/3;
         width:min(100%, calc((100vh - 270px) * 4 / 3)); min-width:min(100%, 320px); margin:0 auto; }  /* buttons stay in view */
.video img { width:100%; height:100%; object-fit:contain; display:block; }
.over { position:absolute; inset:0; display:none; align-items:center; justify-content:center; text-align:center;
        padding:20px; background:rgba(0,0,0,.7); font-size:18px; }
.over.show { display:flex; }
.row { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
button { font:inherit; color:var(--text); background:#232836; border:1px solid var(--line); border-radius:8px;
         padding:10px 14px; cursor:pointer; user-select:none; touch-action:none; }
button:hover { border-color:#4a5368; } button.held { background:#2c3344; transform:translateY(1px); }
.k { display:inline-block; min-width:18px; margin-right:6px; padding:0 5px; border-radius:4px; background:#0f1115;
     color:var(--dim); font-size:12px; text-align:center; }
button.danger { color:#ff9a9a; }
h2 { font-size:12px; text-transform:uppercase; letter-spacing:.8px; color:var(--dim); margin:0 0 8px; font-weight:600; }
.big { font-size:32px; font-weight:700; }
.bar { height:6px; background:#0f1115; border-radius:3px; overflow:hidden; margin:6px 0 10px; }
.bar > div { height:100%; width:0; }
.dim { color:var(--dim); font-size:13px; }
.stage { margin-top:10px; padding:8px 10px; border-radius:8px; background:#0f1115; font-size:14px; }
aside { display:flex; flex-direction:column; gap:16px; min-width:0; }
.crop { background:#000; border-radius:8px; min-height:140px; display:flex; align-items:center; justify-content:center; overflow:hidden; }
.crop img { max-width:100%; max-height:220px; display:none; }
.ex { display:grid; grid-template-columns:96px 1fr 40px; align-items:center; gap:8px; margin:6px 0; font-size:14px; }
.ex .bar { margin:0; height:8px; } .ex b { text-align:right; }
.log { font:12px/1.45 Consolas, monospace; color:var(--dim); max-height:240px; overflow:auto; white-space:pre-wrap; }
#toast { position:fixed; left:50%; top:60px; transform:translateX(-50%); pointer-events:none; background:#232836; border:1px solid var(--line);
         padding:10px 16px; border-radius:8px; display:none; max-width:90vw; }
#toast.show { display:block; } #toast.warn { border-color:var(--warn); color:var(--warn); }
#down { display:none; background:#5a2d2d; padding:10px 16px; } #down.show { display:block; }
body.down main, body.down .chip { opacity:.35; pointer-events:none; } body.down #toast { display:none; }
</style>
</head>
<body>
<div id="down">The detector is not running. Start <b>detector.py</b> in Thonny - this page reconnects by itself.</div>
<header>
  <h1>SmartSort</h1>
  <span class="chip" id="c-cam">Camera</span>
  <span class="chip" id="c-pico">Pico</span>
  <span class="chip" id="c-wifi">Wi-Fi</span>
  <span class="chip" id="c-fps">-</span>
</header>
<main>
  <section class="card">
    <div class="video"><img id="view" alt=""><div class="over" id="over"></div></div>
    <div class="dim" style="margin-top:6px">Hold paper, plastic or electronics in the red box. No teaching needed: the AI already knows them.</div>
    <div class="row">
      <button data-key="e"><span class="k">E</span>Learn the empty view</button>
      <button data-key="b"><span class="k">B</span>Light: <span id="light">-</span></button>
      <button data-key="s"><span class="k">S</span>Snapshot</button>
      <button data-key="q" class="danger" data-confirm="Stop the detector?"><span class="k">Q</span>Quit</button>
    </div>
  </section>
  <aside>
    <div class="card">
      <h2>The AI sees</h2>
      <div class="big" id="live">-</div>
      <div class="bar"><div id="conf"></div></div>
      <div class="dim">Bins: <b id="bins">-</b> <span id="binsnote"></span></div>
      <div class="stage" id="stage">-</div>
    </div>
    <div class="card">
      <h2>What the AI looks at</h2>
      <div class="crop"><img id="crop" alt=""><span class="dim" id="nocrop">nothing yet</span></div>
    </div>
    <div class="card"><h2>How well each kind fits</h2><div id="scores"></div><div class="dim" id="knows"></div></div>
    <div class="card"><h2>Log</h2><div class="log" id="log"></div></div>
  </aside>
</main>
<div id="toast"></div>
<script>
const $ = id => document.getElementById(id);
const COLORS = {paper: 'var(--paper)', plastic: 'var(--plastic)', electronics: 'var(--electronics)', nothing: 'var(--nothing)'};
const NAMES = {paper: 'PAPER', plastic: 'PLASTIC', electronics: 'ELECTRONICS', nothing: 'nothing'};
const WARN = ['Again', 'Empty view used'];
const btn = k => document.querySelector('button[data-key="' + k + '"]');
let up = false, lastLog = '';

function send(key) {
  // A press that can't be delivered within 1.5 s is dropped, so it is never saved later by surprise
  return fetch('/api/key', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-SmartSort': '1'},
                            body: JSON.stringify({key}), signal: AbortSignal.timeout(1500)}).catch(() => {});
}

// Buttons: Quit asks first
document.querySelectorAll('button[data-key]').forEach(b => {
  const k = b.dataset.key;
  b.addEventListener('click', () => { if (!b.dataset.confirm || confirm(b.dataset.confirm)) send(k); });
});

// The same keys as in the detector window
document.addEventListener('keydown', e => {
  if (e.ctrlKey || e.altKey || e.metaKey || e.repeat || e.key.length !== 1) return;
  const k = e.key.toLowerCase();
  if ('ebs'.includes(k)) send(k);
  else if (k === 'q' && confirm(btn(k).dataset.confirm)) send(k);
});

function chip(id, ok, text) {
  const c = $(id); c.textContent = text; c.className = 'chip ' + (ok ? 'ok' : 'bad');
}

function render(s) {
  chip('c-cam', s.camera.ok, s.camera.ok ? 'Camera OK' : 'Camera: ' + (s.camera.frozen_s !== null ? 'no new pictures for ' + Math.round(s.camera.frozen_s) + ' s' : s.camera.status));
  chip('c-pico', s.pico.online, s.pico.online ? 'Pico connected' + (s.pico.message ? ' - ' + s.pico.message : '') : 'Pico NOT connected - ' + s.pico.error);
  chip('c-wifi', !s.wrong_wifi, 'Wi-Fi: ' + (s.wifi || '?'));
  $('c-fps').textContent = s.camera.fps.toFixed(1) + ' pictures/s';

  let over = '';
  if (!s.camera.ok) {
    if (s.wrong_wifi) over = "This PC is on '" + s.wifi + "' - the camera only works on AiPi_Cam_D200";
    else if (s.camera.problem === 'busy') over = 'The camera refuses the picture (busy). Close other pages or phones showing the camera. Still busy after 30 s? Unplug and replug the Pico.';
    else over = 'No new pictures - ' + s.camera.status;
  } else if (s.stage === 'learning') {
    over = 'Learning the empty view - keep it EMPTY and the camera STILL (' + s.empty_progress[0] + '/' + s.empty_progress[1] + ')';
  }
  $('over').textContent = over; $('over').classList.toggle('show', !!over);

  const L = s.live.label, pic = s.camera.ok;   // no new pictures: the old answer means nothing
  $('live').textContent = !pic ? 'no picture' : s.stage === 'learning' ? '...' : NAMES[L] + '  ' + Math.round(s.live.confidence * 100) + '%';
  $('live').style.color = pic ? COLORS[L] : 'var(--dim)';
  $('conf').style.width = pic ? Math.round(s.live.confidence * 100) + '%' : '0';
  $('conf').style.background = COLORS[L];
  $('bins').textContent = NAMES[s.bins]; $('bins').style.color = COLORS[s.bins];
  $('binsnote').textContent = '(an answer must hold ' + s.confirm_s + ' s before a bin opens)';
  $('stage').textContent = !pic ? 'No new pictures from the camera - no bin opens'
    : s.stage === 'learning' ? 'Learning the empty view'
    : s.stage === 'follow' ? (s.back_in_s !== null ? 'Following - nothing seen, back to the red box in ' + s.back_in_s + ' s' : 'Following what you hold')
    : 'Red box: show it paper, plastic or electronics inside the red box';

  $('scores').innerHTML = '';
  for (const lab of ['paper', 'plastic', 'electronics', 'nothing']) {
    const v = pic && s.stage !== 'learning' ? Math.round(s.scores[lab] * 100) : 0, row = document.createElement('div');
    row.className = 'ex';
    row.innerHTML = '<span></span><div class="bar"><div></div></div><b></b>';
    row.children[0].textContent = lab;
    row.children[1].firstChild.style.width = v + '%';
    row.children[1].firstChild.style.background = COLORS[lab];
    row.children[2].textContent = v + '%';
    $('scores').appendChild(row);
  }
  $('knows').textContent = s.ai.error ? 'AI NOT working: ' + s.ai.error
    : 'Preset library (CLIP), no teaching. A bin opens at ' + Math.round(s.min_confidence * 100) + '% or more for ' + s.confirm_s + ' s.';
  $('light').textContent = s.light;

  const log = s.log.slice().reverse().join('\n');
  if (log !== lastLog) { $('log').textContent = log; lastLog = log; }
  const t = $('toast');
  t.textContent = s.message; t.classList.toggle('show', !!s.message);
  t.classList.toggle('warn', WARN.some(w => s.message.startsWith(w)));
}

// The video only streams while this tab is visible: browsers allow just 6 connections to the
// detector, shared by ALL tabs, and each video keeps one open
let streaming = false;
function video(on) {
  if (on === streaming) return;
  streaming = on;
  if (on) $('view').src = '/api/view.mjpg?t=' + Date.now(); else $('view').removeAttribute('src');
}
document.addEventListener('visibilitychange', () => video(up && !document.hidden));

async function poll() {
  try {
    const r = await fetch('/api/status', {cache: 'no-store', signal: AbortSignal.timeout(3000)});
    const s = await r.json();
    const alive = s.running && Date.now() / 1000 - s.time < 5;
    if (alive && !up) streaming = false;   // the detector (re)started: open the video again
    up = alive;
    video(up && !document.hidden);
    $('down').classList.toggle('show', !alive);
    document.body.classList.toggle('down', !alive);
    if (alive) render(s);
  } catch (e) {
    up = false; video(false); $('down').classList.add('show'); document.body.classList.add('down');
  }
  setTimeout(poll, document.hidden ? 1000 : 250);
}

// What the AI looks at: a new picture 3 times a second
const crop = $('crop');
crop.onload = () => { crop.style.display = 'block'; $('nocrop').style.display = 'none'; };
crop.onerror = () => { crop.style.display = 'none'; $('nocrop').style.display = 'inline'; };
setInterval(() => { if (up && !document.hidden) crop.src = '/api/crop.jpg?t=' + Date.now(); }, 330);
poll();
</script>
</body>
</html>
"""


def main():
    global BRIGHTEN
    camera = Camera(CAMERA_STREAM)
    pico = Pico(PICO_PORT)
    wifi = WifiWatch()
    log("Detector started - loading the AI (CLIP, a few seconds)...")
    ai = WasteAI()
    if ai.error:
        log("AI NOT working: %s - no bin will open" % ai.error)
    else:
        log("AI: CLIP preset library for %s - no teaching needed" % ", ".join(WASTE))
    thinker = Thinker(ai)
    dash = Dashboard(DASHBOARD_PORT)
    dash.camera = camera
    if dash.port is not None and OPEN_DASHBOARD:
        dash.open_browser_later()

    recent = []                 # (time, answer) of the AI's answers from the last CONFIRM_SECONDS
    last_id = -1
    scores = dict(NO_SCORES)    # how well each kind fits, in the AI's newest answer
    debug = False
    show_delayed = False        # L: show the delayed answer the bins use instead of the live one
    full_screen = False
    label, conf = "nothing", 0.0
    message, message_until = "", 0
    # Every start: learn the empty view from the first EMPTY_EXAMPLES new pictures
    learn_empty, empty_pictures, empty_tries = True, [], 0
    small, found = None, None   # the newest small picture and what the AI focuses on in it
    # Red-box state: follow is None. Follow state (after START_AFTER pictures of paper, plastic or
    # electronics): follow = where the held thing is; back to the red box after IDLE_AFTER_S of nothing
    follow, waste_in_row, last_waste = None, 0, 0.0
    arrived = collections.deque(maxlen=100)   # when the newest pictures arrived (pictures per second)
    WHITE, YELLOW, GREEN, RED = (255, 255, 255), (0, 220, 255), (80, 220, 80), (80, 80, 240)
    COLORS = {"paper": (255, 230, 180), "plastic": (255, 160, 60), "electronics": (60, 200, 60), "nothing": RED}

    # A big window that fits the screen; you can also drag it bigger or press F
    width = view_width() if SHOW_WINDOW else 960
    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, width, width * 3 // 4)
    sized_for = None
    shown, shown_id, shown_level = None, -1, None
    dash_sent = None                          # the picture the dashboard got last

    try:
        while True:
            frame = camera.frame
            frozen = time.time() - camera.last_time
            wifi.frozen = frozen > 3
            wrong_wifi = wifi.name is not None and "AiPi" not in wifi.name
            if frame is None:
                view = np.zeros((width * 3 // 4, width, 3), np.uint8)
                put(view, "Waiting for camera... (%s)" % camera.status[:45], 30, YELLOW)
                if wrong_wifi:
                    put(view, "This PC is on '%s' - connect it to AiPi_Cam_D200" % wifi.name[:25], 60, RED)
                elif camera.problem == "busy":
                    put(view, BUSY_HELP_1 % (time.time() - camera.problem_since), 60, RED)
                    put(view, BUSY_HELP_2, 85, RED, 0.5)
                else:
                    put(view, "Is this PC on the AiPi_Cam_D200 Wi-Fi?", 60, WHITE)
            else:
                # Only think when a NEW picture has arrived
                if camera.frame_id != last_id:
                    last_id = camera.frame_id
                    arrived.append(time.time())
                    small = small_copy(frame)
                    if learn_empty:
                        empty_pictures.append(small)
                        if len(empty_pictures) >= EMPTY_EXAMPLES:
                            ok, why = check_empty_view(empty_pictures)
                            empty_tries += 1
                            if ok or empty_tries >= 3:   # after 3 tries use it anyway, so you are never stuck
                                ai.set_view(np.median(np.stack(empty_pictures), axis=0).astype(np.uint8))
                                learn_empty = False
                                log("Learned the empty view (%s)%s" % (why, "" if ok else " - used anyway after 3 tries"))
                                if not ok:
                                    message, message_until = "Empty view used anyway: " + why, time.time() + 5
                            else:
                                log("Empty view not good, trying again: " + why)
                                message, message_until = "Again: " + why, time.time() + 3
                            empty_pictures = []
                        label, conf, found, scores = "nothing", 0.0, None, dict(NO_SCORES)
                        recent.clear()
                        pico.item = "nothing"      # no bin opens while it learns the empty view
                    else:
                        found = ai.focus(small, follow)
                        thinker.give(frame, found)   # the AI looks at it in the background (about 0.2 s)

                # A new answer from the AI (it may have skipped pictures while it was busy)
                answer = thinker.take()
                if answer is not None and not learn_empty:
                    label, conf, scores, seen = answer
                    # It SEES something once START_AFTER answers in a row say paper/plastic/electronics
                    # (one odd answer neither starts nor keeps the follow state)
                    waste_in_row = waste_in_row + 1 if label in WASTE else 0
                    if follow is None:
                        if waste_in_row >= START_AFTER:
                            follow, last_waste = seen[0], time.time()   # the red box goes away
                            log("Saw %s - following it" % NAMES[label].lower())
                    else:
                        if waste_in_row >= START_AFTER:
                            last_waste = time.time()
                        if seen[3] == "object":
                            follow = seen[0]           # keep following it (also while it says nothing)
                        if time.time() - last_waste > IDLE_AFTER_S:
                            follow = None
                            log("Nothing for %d s - back to the red box" % IDLE_AFTER_S)
                    now = time.time()
                    # The bins only count answers the AI is sure enough of: a weak guess counts as "nothing"
                    recent.append((now, label if conf >= MIN_CONFIDENCE else "nothing"))
                    while now - recent[0][0] > CONFIRM_SECONDS:
                        recent.pop(0)
                    # Only decide once we have looked for the whole wait
                    if now - recent[0][0] >= CONFIRM_SECONDS - 0.2:
                        best, count = collections.Counter(a for _, a in recent).most_common(1)[0]
                        if count / len(recent) >= CONFIRM_SHARE:
                            pico.item = best   # the bins only act on an answer that held for the whole wait

                if frozen > 3:
                    pico.item = "nothing"   # an old picture must not keep a bin open

                height = width * frame.shape[0] // frame.shape[1]
                if sized_for != (width, height):
                    if SHOW_WINDOW:
                        cv2.resizeWindow(WINDOW, width, height)   # match the camera's picture shape
                    sized_for = (width, height)
                # Brighten and enlarge only when a new picture arrived (or B was pressed)
                if shown_id != camera.frame_id or shown_level != BRIGHTEN:
                    shown = cv2.resize(brighten(frame), (width, height), interpolation=cv2.INTER_LINEAR)
                    shown_id, shown_level = camera.frame_id, BRIGHTEN
                view = shown.copy()
                bottom = height * 640 / width   # the bottom edge, in the 640-wide positions put() uses
                k = width / AI_W
                thick = max(2, width // 320)
                if follow is None:
                    # Red-box state: show it something in here (changes outside it are ignored)
                    cv2.rectangle(view, (int(_MX0 * k), int(_MY0 * k)), (int(_MX1 * k), int(_MY1 * k)), RED, thick)
                if (found is not None and not learn_empty and frozen <= 3
                        and not (found[3] == "middle" and label == "nothing")):
                    # What the AI looks at now: the new thing in the red box, or the thing it follows
                    (x0, y0, x1, y1) = found[0]
                    cv2.rectangle(view, (int(x0 * k), int(y0 * k)), (int(x1 * k), int(y1 * k)),
                                  COLORS[label] if label != "nothing" else WHITE, thick)
                if dash.watched() and dash_sent != (shown_id, shown_level):
                    # The dashboard gets each new picture with only the boxes (the page writes the words itself)
                    dash_sent = (shown_id, shown_level)
                    dash.set_view(view)
                    dash.set_crop(frame, None if learn_empty else found)
                if follow is not None and frozen <= 3:
                    if label in WASTE:
                        put(view, "Following it - the red box comes back after %d s of nothing" % IDLE_AFTER_S,
                            bottom - 105, WHITE, 0.5)
                    else:
                        put(view, "Nothing seen - back to the red box in %d s"
                            % math.ceil(max(0, IDLE_AFTER_S - (time.time() - last_waste))), bottom - 105, WHITE, 0.6)

                if frozen > 3:
                    put(view, "NO NEW PICTURES", 35, RED, 1.0)
                elif learn_empty:
                    put(view, "Learning the empty view - keep it EMPTY, camera STILL (%d/%d)"
                        % (len(empty_pictures), EMPTY_EXAMPLES), 35, YELLOW, 0.7)
                elif show_delayed:
                    # L: the answer the bins act on (it has to hold for about 2 seconds)
                    put(view, "%s  (bins, after %d s)" % (NAMES[pico.item], round(CONFIRM_SECONDS)), 35, COLORS[pico.item], 1.0)
                else:
                    # The AI's answer for THIS picture, without waiting
                    put(view, "%s  %d%%" % (NAMES[label], conf * 100), 35, COLORS[label], 1.0)
                if ai.error:
                    put(view, "AI NOT working - no bin opens: " + ai.error[:50], 65, RED)
                else:
                    put(view, "AI: CLIP preset library - paper, plastic, electronics (no teaching)", 65, WHITE)
                put(view, "Fits: " + "  ".join("%s %d%%" % (lbl, scores[lbl] * 100) for lbl in LABELS)
                    + "   (a bin needs %d%%)" % (MIN_CONFIDENCE * 100), 90, WHITE)
                if frozen > 3 and wrong_wifi:
                    # The PC left the camera's Wi-Fi (switched by hand, or Windows jumped)
                    put(view, "This PC is on '%s' - the camera only works on AiPi_Cam_D200" % wifi.name[:25], 115, RED)
                    put(view, "Switch the Wi-Fi back to AiPi_Cam_D200 - the picture comes back by itself", 140, RED, 0.5)
                elif frozen > 3 and camera.problem == "busy":
                    put(view, BUSY_HELP_1 % frozen, 115, RED)
                    put(view, BUSY_HELP_2, 140, RED, 0.5)
                elif frozen > 3:
                    put(view, "CAMERA FROZEN %ds - reconnecting by itself (%s)" % (frozen, camera.status[:30]), 115, RED)
                    put(view, "If it stays frozen: is the camera powered? Details in detector_log.txt", 140, RED, 0.5)
                elif pico.online:
                    put(view, "Pico (USB): connected   %s" % pico.message[:30], 115, GREEN)
                else:
                    put(view, "Pico NOT connected", 115, RED)
                    put(view, pico.error[:80], 140, RED, 0.5)
                if debug:
                    show_debug(view, frame, ai, found)
                if time.time() < message_until:
                    put(view, message, bottom - 70, GREEN, 0.8)
                put(view, "L label: %s    E empty view    D debug"
                    % ("bins (after %d s)" % round(CONFIRM_SECONDS) if show_delayed else "live"), bottom - 37, WHITE, 0.5)
                put(view, "B light: %s    F full screen    S snapshot    Q quit"
                    % BRIGHTEN_NAMES[BRIGHTEN], bottom - 12, WHITE, 0.5)

            # Everything the dashboard shows (http://localhost:8000/api/status)
            now = time.time()
            dash.status = {
                "running": True,
                "time": now,
                "camera": {"ok": frame is not None and frozen <= 3, "status": camera.status, "problem": camera.problem,
                           "frozen_s": round(frozen, 1) if frame is not None else None,
                           "fps": sum(1 for t in arrived if t > now - 2) / 2},
                "wifi": wifi.name,
                "wrong_wifi": wrong_wifi,
                "pico": {"online": pico.online, "message": pico.message, "error": pico.error},
                "stage": "learning" if learn_empty else ("follow" if follow is not None else "redbox"),
                "empty_progress": [len(empty_pictures), EMPTY_EXAMPLES],
                "back_in_s": (math.ceil(max(0, IDLE_AFTER_S - (now - last_waste)))
                              if follow is not None and label not in WASTE else None),
                "live": {"label": label, "confidence": round(conf, 3)},
                "bins": pico.item,
                "confirm_s": CONFIRM_SECONDS,
                "focus": (None if found is None or learn_empty else
                          {"box": [round(found[0][0] / AI_W, 3), round(found[0][1] / AI_H, 3),
                                   round(found[0][2] / AI_W, 3), round(found[0][3] / AI_H, 3)], "kind": found[3]}),
                "scores": scores,
                "min_confidence": MIN_CONFIDENCE,
                "ai": {"model": "CLIP clip-vit-base-patch32 (preset library, no teaching)",
                       "error": ai.error, "look_s": round(thinker.look_s, 2)},
                "message": message if now < message_until else "",
                "light": BRIGHTEN_NAMES[BRIGHTEN],
                "log": list(LOG_LINES)[-25:],
            }
            dash.tick()

            if SHOW_WINDOW:
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(30) & 0xFF
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break   # the window was closed with its X button
            else:
                time.sleep(0.03)
                key = 255
            if key == 255:
                key = dash.next_key()   # a button on the dashboard
            if key == ord("q"):
                break
            elif key == ord("b"):
                BRIGHTEN = (BRIGHTEN + 1) % 3   # only the picture on screen
                message, message_until = "Brightening: %s" % BRIGHTEN_NAMES[BRIGHTEN], time.time() + 1.5
                print(message)
            elif key == ord("f") and SHOW_WINDOW:
                full_screen = not full_screen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if full_screen else cv2.WINDOW_NORMAL)
            elif key == ord("l"):
                show_delayed = not show_delayed
                message = ("Big label: the answer the bins use (after %d s)" % round(CONFIRM_SECONDS)
                           if show_delayed else "Big label: the live answer")
                message_until = time.time() + 1.5
            elif key == ord("d"):
                debug = not debug
            elif key == ord("s") and frame is not None:
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                path = os.path.join(SNAPSHOT_DIR, "%d.jpg" % (time.time() * 1000))
                cv2.imwrite(path, frame)
                print("Saved snapshot:", path)
                message, message_until = "Snapshot saved: snapshots\\" + os.path.basename(path), time.time() + 2
            elif key == ord("e"):
                learn_empty, empty_pictures, empty_tries = True, [], 0
                follow, waste_in_row = None, 0
                message, message_until = "Learning the empty view again - keep it empty", time.time() + 2
                log("Learning the empty view again (E)")
    finally:
        # Also after an error: close the bins, free the dashboard's port, the camera and the USB cable
        pico.item = "nothing"
        log("Detector stopped")
        dash.close()
        time.sleep(0.4)   # let the last "nothing" reach the Pico
        camera.stop = pico.stop = True
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
