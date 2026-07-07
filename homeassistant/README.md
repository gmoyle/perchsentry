# BirdBuddy — Home Assistant integration

A custom integration that pulls your BirdBuddy camera into Home Assistant: a
live camera feed, sensors for what was last seen and how the Pi is doing, and
events you can hang notifications off. Local polling only — nothing leaves your
network.

## What you get

- **`camera.birdbuddy_camera`** — the live MJPEG feed, with `/snapshot` stills
  for notifications and picture cards.
- **Sensors** — last bird, last animal, last confidence, today's sightings,
  animals today, CPU temperature.
- **Binary sensors** — `Bird detected` / `Animal detected`, on for a minute
  after each detection (handy for dashboards and simple automations).
- **Events** — `birdbuddy_bird_detected` and `birdbuddy_animal_detected` fire
  the instant a new detection is polled, carrying the species, confidence, and a
  URL to the captured photo. This is the clean hook for notifications.

## Install

**Manual (simplest):**

1. Copy the `custom_components/birdbuddy` folder from this repo into your Home
   Assistant config directory, so you end up with
   `<config>/custom_components/birdbuddy/`.
2. Restart Home Assistant.
3. **Settings → Devices & Services → Add Integration → BirdBuddy.**
4. Enter your BirdBuddy host (e.g. `birdbuddy.local`) and port (`8080`).

That's it — a BirdBuddy device appears with the camera and sensors.

## Notifications

BirdBuddy fires an event on every new detection. Point a notification at it —
this example sends the photo to your phone whenever a bird is identified:

```yaml
automation:
  - alias: "BirdBuddy — bird notification"
    trigger:
      - platform: event
        event_type: birdbuddy_bird_detected
    action:
      - service: notify.mobile_app_your_phone   # change to your device
        data:
          title: "🐦 {{ trigger.event.data.species }}"
          message: >
            Spotted at the feeder
            ({{ (trigger.event.data.confidence * 100) | round }}% sure)
          data:
            image: "{{ trigger.event.data.image_url }}"
```

Swap `birdbuddy_bird_detected` for `birdbuddy_animal_detected` to be pinged
about squirrels, deer, and the rest. Both events carry the same fields:
`kind`, `species`, `confidence` (0–1), and `image_url`.

## A dashboard card

A picture-glance card gives you the live feed with the latest sighting under it:

```yaml
type: picture-glance
title: Feeder
camera_image: camera.birdbuddy_camera
camera_view: live
entities:
  - sensor.birdbuddy_last_bird
  - sensor.birdbuddy_today_s_sightings
  - sensor.birdbuddy_cpu_temperature
```

## Notes

- This talks to BirdBuddy's local HTTP API (`/api/ha`, `/stream`, `/snapshot`);
  no cloud, no account.
- It polls every few seconds, so a notification lands within a few seconds of
  the bird — plenty quick for a feeder, and it keeps the setup to "add the
  integration, enter the address" with nothing to configure on the Pi.
- The existing ntfy notifications in BirdBuddy still work; this doesn't touch
  them. Use whichever you prefer, or both.
