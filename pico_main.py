# SmartSort - Raspberry Pi Pico W
# Three bins, one servo each. The detector (running on your PC) says what it sees,
# and the matching bin opens: paper, plastic or electronics.
# The detector talks to the Pico over the USB cable (lines like "item plastic").
# Save this on the Pico as main.py so it starts by itself.
#
# The Pico's own green LED (next to the USB plug):
#   slow blinking = waiting for the detector,  on = the detector is talking to it
#
# PHONE REMOTE (WIFI = "remote"): the Pico makes its own Wi-Fi called SmartSort.
# Join it with your phone and open http://192.168.4.1 to open and close every bin.
# It works with or without the detector, and the camera's Wi-Fi stays free for the PC.

import network
import rp2
import sys
import select
import time
import json
import asyncio
import machine
from machine import Pin, PWM, UART

# ---------------- Settings ----------------
# Wi-Fi and the dashboard (the web page with a button for every bin):
#   "remote" = the Pico makes its OWN Wi-Fi for your phone: dashboard at http://192.168.4.1
#   "camera" = the Pico joins the camera's Wi-Fi: dashboard at http://192.168.169.50
#              (takes one of the camera's few places, but the page can show the camera)
#   None     = no Wi-Fi, only the detector on the USB cable
WIFI = "remote"
REMOTE_NAME = "SmartSort"        # the Wi-Fi your phone joins
REMOTE_PASSWORD = "sortmybins"   # at least 8 characters
REMOTE_OPEN_S = 15               # a bin opened from the phone stays open this long (or press Close)

WIFI_NAME = "AiPi_Cam_D200"      # "camera" only: must match the camera's Wi-Fi name EXACTLY
WIFI_PASSWORD = "12345678"       # ignored automatically if the network has no password
PICO_IP = "192.168.169.50"       # fixed address, so the dashboard link never changes
CAMERA_STREAM = "http://192.168.169.1/stream"

# Each bin: (name, GP pin, closed angle, open angle)
# The lids hinge on their servos (bin_parts\smartsort_bin.py). Closed is 90: push the horn on
# pointing straight down while the Pico holds the servo there (it does at every start).
# Open is 85 degrees away; the lid would only touch the frame at 98.
# If a lid pushes down instead of opening, change its open angle from 5 to 175 (or back).
# If a lid doesn't sit flat when closed, move both its angles the same amount, e.g. 95 and 10.
BINS = [
    ("paper", 16, 90, 175),        # lid 1, servo at its back-right - GP16 (physical pin 21)
    ("plastic", 17, 90, 175),      # lid 2, servo at its back-right - GP17 (physical pin 22)
    ("electronics", 18, 90, 5),    # lid 3, servo at its back-left - GP18 (physical pin 24, after the GND at pin 23)
]
OPEN_TIME_S = 3       # a bin stays open this many seconds after its waste was last seen
SERVO_GAP_MS = 300    # wait between moving two servos, so they never all pull power at once
RELEASE_AFTER_MS = 600  # after closing, the servo's signal is switched off to save power

onboard = Pin("LED", Pin.OUT)   # the Pico W's own LED: is the detector talking to us?
green = Pin(4, Pin.OUT)    # GP4: a bin is open (optional LED)
yellow = Pin(3, Pin.OUT)   # GP3: detector on the PC is not running (optional LED)
red = Pin(2, Pin.OUT)      # GP2: all bins closed (optional LED)

bins = {}
for name, pin, closed, opened in BINS:
    servo = PWM(Pin(pin))
    servo.freq(50)
    servo.duty_ns(0)   # no signal yet: the servo stays still and uses almost no power
    bins[name] = {"servo": servo, "closed": closed, "opened": opened,
                  "open": False, "count": 0, "close_at": 0, "release_ms": None}

# Camera serial: GP20 = Pico TX -> camera RX, GP21 = Pico RX <- camera TX
cam_uart = UART(1, baudrate=2000000, tx=Pin(20), rx=Pin(21), rxbuf=2048)

wlan = network.WLAN(network.STA_IF)   # joining the camera's Wi-Fi (WIFI = "camera")
ap = network.WLAN(network.AP_IF)      # the Pico's own Wi-Fi for your phone (WIFI = "remote")

# ---------------- State ----------------
seeing = "nothing"
last_report_ms = None
fps = None
last_fps_ms = None


def set_angle(servo, angle):
    # 0 degrees = 0.5 ms pulse, 180 degrees = 2.5 ms pulse
    us = 500 + angle * 2000 / 180
    servo.duty_ns(int(us * 1000))


def ago(ms):
    return time.ticks_diff(time.ticks_ms(), ms)


def detector_online():
    return last_report_ms is not None and ago(last_report_ms) < 3000


def update_leds():
    any_open = any(b["open"] for b in bins.values())
    green.value(1 if any_open else 0)
    red.value(0 if any_open else 1)


def close_bin(name):
    b = bins[name]
    b["open"] = False
    set_angle(b["servo"], b["closed"])
    b["release_ms"] = time.ticks_ms()   # bins_task lets go of the servo once it has closed
    update_leds()
    print("%s bin closed" % name)


async def close_all():
    # One servo at a time, so they don't all pull power at the same moment
    for name in bins:
        close_bin(name)
        await asyncio.sleep_ms(SERVO_GAP_MS)


async def open_bin(name, count=True, open_s=OPEN_TIME_S):
    b = bins[name]
    # The bin stays open until close_at. Opening it again can only make that later, never
    # sooner, so the detector's 3 s can't cut short a bin you opened from the phone
    close_at = time.ticks_add(time.ticks_ms(), open_s * 1000)
    if not b["open"] or time.ticks_diff(close_at, b["close_at"]) > 0:
        b["close_at"] = close_at
    if not b["open"]:
        # Only one bin open at a time, so waste never goes in the wrong one
        closed_one = False
        for other in bins:
            if other != name and bins[other]["open"]:
                close_bin(other)
                closed_one = True
        if closed_one:
            await asyncio.sleep_ms(SERVO_GAP_MS)
        b["open"] = True
        b["release_ms"] = None
        if count:
            b["count"] += 1
        set_angle(b["servo"], b["opened"])
        update_leds()
        print("%s! Bin open" % name)


async def report(item):
    # The detector tells us what the camera sees: paper / plastic / electronics / nothing
    global seeing, last_report_ms
    if item == "bio":   # an older detector still says "bio" for this bin
        item = "electronics"
    last_report_ms = time.ticks_ms()
    seeing = item
    if item in bins:
        await open_bin(item)


def get_param(path, key):
    # get_param("/detect?item=paper", "item") -> "paper"
    if "?" not in path:
        return None
    for part in path.split("?", 1)[1].split("&"):
        if part.startswith(key + "="):
            return part[len(key) + 1:]
    return None


# ---------------- USB cable ----------------
async def usb_task():
    # Reads one command per line from the PC over the USB cable, for example "item plastic".
    # Every command is answered with "ok ...", so the detector knows the Pico is listening.
    poll = select.poll()
    poll.register(sys.stdin, select.POLLIN)
    line = ""
    while True:
        while poll.poll(0):
            ch = sys.stdin.read(1)
            if ch in ("\n", "\r"):
                parts = line.split()
                line = ""
                if len(parts) == 2 and parts[0] == "item":
                    print("ok", parts[1])
                    await report(parts[1])
                elif parts == ["ping"]:
                    print("ok ping")
            else:
                line += ch
                if len(line) > 60:   # not a command we know: forget it
                    line = ""
        await asyncio.sleep_ms(20)


# ---------------- Wi-Fi (only when WIFI is "remote" or "camera") ----------------
def start_remote_wifi():
    # WIFI = "remote": the Pico makes its own Wi-Fi for your phone. The phone talks straight
    # to the Pico, so the remote also works while the detector on the PC is off
    if len(REMOTE_PASSWORD) < 8:
        print("REMOTE_PASSWORD needs at least 8 characters - no phone remote this time")
        return
    ap.config(essid=REMOTE_NAME, password=REMOTE_PASSWORD)
    ap.active(True)
    for _ in range(50):   # wait up to 5 s for it to start
        if ap.active():
            break
        time.sleep(0.1)
    if ap.active():
        print("Phone remote: join the Wi-Fi '%s' (password %s), then open http://%s"
              % (REMOTE_NAME, REMOTE_PASSWORD, ap.ifconfig()[0]))
    else:
        print("Could not start the '%s' Wi-Fi - no phone remote this time" % REMOTE_NAME)


def join_wifi():
    # Tries once to join the camera's Wi-Fi (about 20 s). Returns True when connected.
    wlan.active(True)
    wlan.config(pm=0xa11140)   # turn off Wi-Fi power saving, so the Pico answers quickly
    if wlan.isconnected():
        # Still connected (for example after a soft restart): just make sure of the address
        if wlan.ifconfig()[0] != PICO_IP:
            wlan.ifconfig((PICO_IP, "255.255.255.0", "192.168.169.1", "192.168.169.1"))
        onboard.on()
        print("Wi-Fi OK (already connected)! Address: %s" % wlan.ifconfig()[0])
        return True
    # Show every Wi-Fi network the Pico can see, so a wrong name is easy to spot
    print("Looking for Wi-Fi networks...")
    seen = []
    is_open = False
    for net in wlan.scan():
        try:
            name = net[0].decode()
        except UnicodeError:
            continue
        if name == WIFI_NAME and net[4] == 0:
            is_open = True   # security 0 = no password needed
        if name and name not in seen:
            seen.append(name)
            print("  found: %s | channel %d | signal %d" % (name, net[2], net[3]))
    if WIFI_NAME not in seen:
        print("'%s' was NOT found! Is the camera on? Check that WIFI_NAME matches exactly." % WIFI_NAME)
        return False
    print("Connecting to", WIFI_NAME, "...")
    if is_open:
        wlan.connect(WIFI_NAME)
    else:
        wlan.connect(WIFI_NAME, WIFI_PASSWORD)
    for _ in range(40):   # wait up to 20 s, blinking the LED
        if wlan.isconnected():
            break
        onboard.toggle()
        time.sleep(0.5)
    if not wlan.isconnected():
        # status: -3 = wrong password, -2 = network not found, -1 = failed
        print("Could not join %s (status %d) - trying again..." % (WIFI_NAME, wlan.status()))
        wlan.disconnect()
        onboard.off()
        return False
    # Use a fixed address so the dashboard link is always the same
    wlan.ifconfig((PICO_IP, "255.255.255.0", "192.168.169.1", "192.168.169.1"))
    onboard.on()
    print("Wi-Fi OK! Address: %s  Dashboard: http://%s" % (wlan.ifconfig()[0], PICO_IP))
    return True


def join_wifi_or_restart():
    # Keeps trying to join. The camera and the Pico switch on at the same moment,
    # and the camera's Wi-Fi needs a few seconds to start, so the first tries often fail.
    # If it still fails after about 2 minutes, restart the Pico (the Wi-Fi chip can get stuck).
    for attempt in range(6):
        if join_wifi():
            return
        time.sleep(2)
    print("Still no Wi-Fi - restarting the Pico...")
    time.sleep(1)
    machine.reset()


async def wifi_task():
    # Rejoins the camera's Wi-Fi by itself if it drops (for example when the camera restarts)
    while True:
        if not wlan.isconnected():
            onboard.off()
            print("Wi-Fi lost! Closing the bins and rejoining...")
            await close_all()
            join_wifi_or_restart()
        await asyncio.sleep(2)


# ---------------- Background jobs ----------------
async def bins_task():
    # Closes a bin when its time is up: OPEN_TIME_S after its waste was last seen,
    # or REMOTE_OPEN_S after you opened it from the phone
    tick = 0
    while True:
        tick += 1
        if detector_online():
            onboard.on()                     # the detector is talking to us
        elif tick % 5 == 0:
            onboard.toggle()                 # slow blink: waiting for the detector
        for name, b in bins.items():
            if b["open"] and time.ticks_diff(time.ticks_ms(), b["close_at"]) > 0:
                close_bin(name)
            # A closed lid just rests, so stop the signal: otherwise a servo pressing
            # against its end stop keeps pulling current (and hums)
            if b["release_ms"] is not None and ago(b["release_ms"]) > RELEASE_AFTER_MS:
                b["servo"].duty_ns(0)
                b["release_ms"] = None
        yellow.value(0 if detector_online() else 1)
        await asyncio.sleep_ms(100)


async def camera_task():
    # Reads the camera's "FPS dvp_mjpeg:  6," lines over the wires
    global fps, last_fps_ms
    reader = asyncio.StreamReader(cam_uart)
    while True:
        line = await reader.readline()
        try:
            text = line.decode()
        except UnicodeError:
            continue
        if "FPS" in text:
            num = ""
            for ch in text.split(":", 1)[-1]:
                if ch.isdigit():
                    num += ch
                elif num:
                    break
            if num:
                fps = int(num)
                last_fps_ms = time.ticks_ms()


# ---------------- Dashboard page ----------------
PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartSort</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--line:#262a33;--text:#e8eaf0;--muted:#8b92a3;--g:#22c55e;--r:#ef4444;--y:#eab308}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text)}
header{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--line)}
h1{font-size:20px;margin:0}h1 span{color:var(--g)}
.pill{font-size:13px;padding:4px 10px;border-radius:99px;background:#222631;color:var(--muted)}
.pill.ok{background:rgba(34,197,94,.15);color:var(--g)}.pill.bad{background:rgba(239,68,68,.15);color:var(--r)}
main{display:grid;gap:16px;padding:16px;max-width:1100px;margin:0 auto}
@media(min-width:820px){main{grid-template-columns:1.6fr 1fr}}
main.nocam{grid-template-columns:1fr;max-width:520px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
.cam{position:relative;padding:0;overflow:hidden;aspect-ratio:4/3;background:#000}
.cam img{width:100%;height:100%;object-fit:contain;display:block}
.msg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--muted);text-align:center}
h2{font-size:13px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin:0 0 12px}
.bins{display:grid;gap:10px}
.bin{display:flex;align-items:center;justify-content:space-between;gap:10px;border:2px solid var(--line);border-radius:12px;padding:12px 14px}
.bin b{font-size:20px}.bin small{display:block;font-size:13px;color:var(--muted);margin-top:2px}
.bin.open{border-color:var(--g);background:rgba(34,197,94,.15)}.bin.open b{color:var(--g)}
button{padding:10px 16px;border:0;border-radius:12px;background:#2a2f3a;color:var(--text);font-size:15px;font-weight:600}
.wide{width:100%;margin-top:10px}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:16px}
.tile{background:#11141a;border:1px solid var(--line);border-radius:12px;padding:10px}
.tile span{display:block;font-size:12px;color:var(--muted)}
.tile b{font-size:22px}
.yes{color:var(--g)}.no{color:var(--muted)}.warn{color:var(--y)}
</style></head>
<body>
<header><h1>Smart<span>Sort</span></h1><span id="conn" class="pill">connecting...</span></header>
<main>
  <section class="card cam">
    <img id="cam" alt="Camera" hidden>
    <div id="camnote" class="msg"><div>The camera can send its picture to only ONE viewer.<br>
      While the detector runs on the PC, leave this off.<br><br><button id="camon">Show camera here</button></div></div>
    <div id="camerr" class="msg" hidden>Camera stream not available.<br>Retrying...</div>
  </section>
  <section class="card">
    <h2>Bins</h2>
    <div id="bins" class="bins"></div>
    <button id="closeall" class="wide">Close all</button>
    <div class="tiles">
      <div class="tile"><span>Camera sees</span><b id="seeing">-</b></div>
      <div class="tile"><span>Total sorted</span><b id="total">-</b></div>
      <div class="tile"><span>Detector (PC)</span><b id="det">-</b></div>
      <div class="tile"><span>Camera FPS</span><b id="fps">-</b></div>
    </div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const STREAM = "STREAM_URL";
const ORDER = ['paper', 'plastic', 'electronics'];
const NAMES = {paper: 'Paper', plastic: 'Plastic', electronics: 'Electronics', nothing: 'Nothing'};
$('bins').innerHTML = ORDER.map(n =>
  `<div class="bin" id="bin-${n}"><div><b>${NAMES[n]}</b><small id="st-${n}">-</small></div><button id="btn-${n}" data-bin="${n}">Open</button></div>`
).join('');
let last = null;   // the newest status, so a bin's button knows whether it opens or closes
$('bins').onclick = e => {
  const n = e.target.dataset.bin;
  if (n) call((last && last.bins[n].open ? '/close?bin=' : '/open?bin=') + n);
};
if (!STREAM) {   // the phone is on the Pico's own Wi-Fi: it can't reach the camera
  document.querySelector('.cam').hidden = true;
  document.querySelector('main').classList.add('nocam');
}
function show(d){
  last = d;
  let total = 0;
  for (const n of ORDER) {
    const b = d.bins[n];
    total += b.count;
    $('bin-' + n).className = 'bin' + (b.open ? ' open' : '');
    $('btn-' + n).textContent = b.open ? 'Close' : 'Open';
    $('st-' + n).textContent = (b.open ? 'OPEN' : 'Closed') + ' - ' + b.count + ' sorted';
  }
  $('total').textContent = total;
  $('seeing').textContent = NAMES[d.seeing] || d.seeing;
  $('seeing').className = d.seeing === 'nothing' ? 'no' : 'yes';
  $('det').textContent = d.detector ? 'Running' : 'Off'; $('det').className = d.detector ? 'yes' : 'warn';
  $('fps').textContent = d.fps === null ? '-' : d.fps;
  $('conn').textContent = 'Pico connected'; $('conn').className = 'pill ok';
}
async function call(url){
  try { const r = await fetch(url, {cache: 'no-store'}); show(await r.json()); }
  catch(e){ $('conn').textContent = 'Pico offline'; $('conn').className = 'pill bad'; }
}
$('closeall').onclick = () => call('/close');
setInterval(() => call('/status'), 500);
call('/status');
const cam = $('cam');
let camOn = false;   // only take the camera's one stream when asked
$('camon').onclick = () => { camOn = true; $('camnote').hidden = true; cam.hidden = false; cam.src = STREAM; };
cam.onerror = () => {
  if (!camOn) return;
  $('camerr').hidden = false;
  setTimeout(() => { cam.removeAttribute('src'); cam.src = STREAM; }, 3000);
};
cam.onload = () => { $('camerr').hidden = true; };
</script>
</body></html>
""".replace("STREAM_URL", CAMERA_STREAM if WIFI == "camera" else "").encode()


# ---------------- Web server ----------------
def status_json():
    fps_fresh = last_fps_ms is not None and ago(last_fps_ms) < 5000
    return json.dumps({
        "bins": {name: {"open": b["open"], "count": b["count"]} for name, b in bins.items()},
        "seeing": seeing if detector_online() else "nothing",
        "detector": detector_online(),
        "fps": fps if fps_fresh else None,
    })


async def send(writer, body, content_type, code="200 OK"):
    if isinstance(body, str):
        body = body.encode()
    writer.write(("HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %d\r\nCache-Control: no-store\r\n"
                  "Connection: close\r\n\r\n" % (code, content_type, len(body))).encode())
    writer.write(body)
    await writer.drain()


async def handle(reader, writer):
    try:
        request = await reader.readline()
        while True:  # skip the request headers
            header = await reader.readline()
            if not header or header == b"\r\n":
                break
        parts = request.decode().split(" ")
        path = parts[1] if len(parts) > 1 else "/"

        if path.startswith("/detect"):
            # An older detector can still report over Wi-Fi: /detect?item=paper / plastic / electronics / nothing
            item = get_param(path, "item")
            if item is None:   # the oldest paper-only detector sends paper=1 / paper=0
                item = "paper" if "paper=1" in path else "nothing"
            await send(writer, status_json(), "application/json")
            await report(item)
        elif path.startswith("/status"):
            await send(writer, status_json(), "application/json")
        elif path.startswith("/open"):
            name = get_param(path, "bin")
            if name in bins:
                await open_bin(name, count=False, open_s=REMOTE_OPEN_S)
            await send(writer, status_json(), "application/json")
        elif path.startswith("/close"):
            name = get_param(path, "bin")   # /close?bin=paper closes one bin, /close closes them all
            if name in bins:
                if bins[name]["open"]:
                    close_bin(name)
            else:
                await close_all()
            await send(writer, status_json(), "application/json")
        elif path == "/" or path.startswith("/?"):
            await send(writer, PAGE, "text/html; charset=utf-8")
        else:
            await send(writer, "Not found", "text/plain", "404 Not Found")
    except Exception as e:
        print("Web error:", e)
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------- Start ----------------
async def main():
    # Start the Wi-Fi FIRST and only then move the servos: Wi-Fi and servos
    # both need a lot of power, and together they can make the camera restart
    if WIFI != "remote":
        ap.active(False)   # a soft restart keeps the Pico's own Wi-Fi running
    if WIFI == "camera":
        join_wifi_or_restart()
    else:
        # Leave the camera's Wi-Fi (a soft restart keeps the old connection),
        # so its places are free for the PC
        wlan.disconnect()
        wlan.active(False)
        if WIFI == "remote":
            start_remote_wifi()
    await close_all()
    asyncio.create_task(bins_task())
    asyncio.create_task(camera_task())
    asyncio.create_task(usb_task())
    if WIFI:
        await asyncio.start_server(handle, "0.0.0.0", 80)
    if WIFI == "camera":
        asyncio.create_task(wifi_task())
        print("Dashboard: http://%s (on the %s Wi-Fi)" % (PICO_IP, WIFI_NAME))
    print("SmartSort ready! Waiting for the detector on the USB cable...")
    while True:
        await asyncio.sleep(3600)


try:
    asyncio.run(main())
finally:
    asyncio.new_event_loop()
