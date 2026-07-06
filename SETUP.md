# BirdBuddy — host setup notes

Most of the app is self-contained in this repo and runs from the
`birdbuddy.service` systemd unit (gunicorn, user `gmoyle`, port 8080).

A couple of pieces live **outside** the repo because they need root and
therefore can't be shipped as plain app files. Recreate them after a rebuild.

## Fan control (Quiet / Silent recording modes)

The "Fan (recording)" setting silences the Pi 5 active cooler during a slow-mo
burst so it doesn't whine into an attached microphone. The gunicorn service
runs as non-root and cannot write the fan sysfs nodes, so it shells out to a
small root helper through a narrow `sudo` rule.

Two out-of-tree files, both recreatable from this repo:

### 1. Root helper — `/usr/local/sbin/birdbuddy-fanctl`

There are **two copies of this program, and they are separate files:**

| File | Role |
|------|------|
| [`fanctl.py`](fanctl.py) (this repo) | The master copy — edit this, git tracks it |
| `/usr/local/sbin/birdbuddy-fanctl` | The installed copy that actually runs as root |

They must be separate: a program run as root has to live somewhere a non-root
user can't modify, so it can't just run out of this user-owned repo folder.

> ⚠️ **Editing `fanctl.py` does NOT change fan behavior until you reinstall it.**
> The running fan control is `/usr/local/sbin/birdbuddy-fanctl`, a copy. After
> any edit to `fanctl.py`, copy it over again:

```bash
sudo install -o root -g root -m 0755 fanctl.py /usr/local/sbin/birdbuddy-fanctl
```

(This same command is also how you create it in the first place on a fresh
machine.) It sets the pwm-fan cooling device's `cur_state` (0=off … 4=max):
`off`→0, `quiet`→1, `auto`→the level the current temperature calls for.

### 2. Sudoers rule — `/etc/sudoers.d/010-birdbuddy-fan`

Grants the service user passwordless access to *only* those three subcommands:

```bash
sudo tee /etc/sudoers.d/010-birdbuddy-fan >/dev/null <<'EOF'
gmoyle ALL=(root) NOPASSWD: /usr/local/sbin/birdbuddy-fanctl off, /usr/local/sbin/birdbuddy-fanctl quiet, /usr/local/sbin/birdbuddy-fanctl auto
EOF
sudo chmod 0440 /etc/sudoers.d/010-birdbuddy-fan
sudo visudo -c -f /etc/sudoers.d/010-birdbuddy-fan   # validate
```

### Verify

```bash
sudo -n /usr/local/sbin/birdbuddy-fanctl quiet   # → 1
sudo -n /usr/local/sbin/birdbuddy-fanctl auto    # → level for current temp
```

### Safety notes

- Fan control drives the kernel cooling device, so the thermal governor still
  reasserts the fan at its trip points; `RecordingFan` also restores `auto`
  when the recording ends and has an in-app 78°C watchdog.
- Firmware CPU throttling (~80°C soft, ~85°C hard, ~110°C shutdown) is
  independent of all this — the SoC is protected regardless of fan mode.

## Temperature monitoring

No host setup required. [`sysmon.py`](sysmon.py) samples the SoC temperature
every 60s to `logs/temps.csv` (48h rolling window, gitignored) and the stats
page charts it via `/api/temp` and `/api/temp-history`.
