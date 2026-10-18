# WebcamXR – Test VR Projects With a Webcam

WebcamXR lets you play and test a VR Template project **without a headset**.
A small Python tracker reads your webcam (head and both hands, with Google
MediaPipe) and streams the data to Unreal over OSC. In Unreal, your head moves
the camera, your hands move the motion controllers, and hand gestures trigger
the same Enhanced Input actions as the controller buttons. Grab, shoot and
teleport all work as they do with a headset, and your game code needs no changes.

When a headset is connected, webcam tracking switches itself off and the
project runs as a normal VR project.

All video is processed locally on your PC. Nothing is uploaded.

## Requirements

- Unreal Engine 5.8 project created from the **Virtual Reality** template
  (WebcamXR builds on its `BP_XRPawn`, input actions and hand meshes)
- **OSC** plugin enabled (Edit → Plugins → OSC, restart the editor)
- Windows 10/11, a webcam
- Python 3 (64-bit) from python.org. Tick "Add python.exe to PATH" when you install it.

## Installation

**From Fab:** add WebcamXR to your VR Template project. It installs into
`Content/WebcamXR` and `Content/Python/WebcamXR` and includes the demo map
`WebcamXR/Maps/L_WebcamXRDemo`.

**From GitHub:** close the editor and copy the repository's `Content/WebcamXR`
and `Content/Python/WebcamXR` folders into your project's `Content` folder.
The GitHub version has no demo map (it would be a copy of Epic's template
level). Use the template's own level instead:

1. Open `XRFramework/Levels/L_XRTemplate`.
2. Drag `WebcamXR/Blueprints/BP_WebcamOSCReceiver` into the level.
3. World Settings → GameMode Override → `BP_WebcamXRGameMode`.
4. Save the level.

Then, for both versions, enable the **OSC** plugin and restart the editor.

## Quick start

1. Double-click `Content/Python/WebcamXR/Start_WebcamTracker.bat`.
   The first run creates a Python environment in
   `%LOCALAPPDATA%\WebcamXR\venv` and downloads the packages. This needs
   internet and takes a few minutes. Later runs start right away.
   A preview window shows what the camera sees.
2. Open the demo map (Fab) or the level you set up above (GitHub) and press **Play**.
3. Hold both hands up, **palms facing the camera, fingers pointing up**, and
   keep them still for about 2 seconds. Your virtual hands appear.

## Using WebcamXR in your own level

- Place one **BP_WebcamOSCReceiver** actor in the level. It listens on UDP
  port 7000 on all network adapters.
- Use **BP_WebcamXRGameMode** as the level's GameMode Override (World
  Settings), or spawn **BP_WebcamXRPawn**. It is a child of the template's
  `BP_XRPawn`, so everything from the template keeps working.
- Using your own pawn instead: add the **BPC_WebcamXR** component and call
  `SetupWebcamXR` on BeginPlay. Pass in the camera, the four motion
  controllers (left/right grip and aim) and the two hand meshes.
  `BP_WebcamXRPawn` is a working example.

If the level has no receiver, webcam tracking turns itself off and prints a
message.

## Gestures

| Gesture | Action |
|---|---|
| Move / turn your hand | Virtual hand and motion controller follow |
| Make a fist | Grab (controller grip) |
| Open your hand (2+ fingers) | Release |
| While holding a gun: extend index finger, then curl it | Shoot (trigger) |
| **Right hand** points (only index finger out), hold ~0.5 s | Teleport arc from the fingertip |
| Stop pointing (open hand or fist) | Teleport to the arc's end |

## Keyboard and mouse (only without a headset)

| Key | Action |
|---|---|
| Left / right mouse button | Grab with left / right hand |
| Space (hold, release) | Teleport |
| Q / E | Snap turn left / right |
| F / G | Shoot right / left hand gun |
| Tab | Left hand menu |
| C | Recalibrate hands |
| R | Recalibrate head |

## Settings

Select the **WebcamXR** component on the pawn → Details → **Webcam Tracking**.
You can change these values during Play as well.

| Setting | Default | Description |
|---|---|---|
| Webcam Tracking Enabled | on | Turns webcam tracking off completely |
| Auto Disable Webcam With HMD | on | Turns webcam tracking off when a headset is active |
| Hand Calibration Delay | 1.75 | Seconds a hand must be visible before it calibrates |
| Left / Right Hand Body Offset | (40, ∓20, -35) | Hand rest position relative to the play-space origin (cm) |
| Left / Right Hand Neutral Rotation | (0, -90, -90) | Hand rotation in the calibration pose |
| Hand Location / Rotation Smoothing | 12 / 14 | Higher = more responsive, more jitter |
| Head Position Sensitivity X/Y/Z | 200 | How far the camera moves with your head |
| Desktop Field Of View | 120 | Camera FOV without a headset (0 = keep the camera's own value) |
| Desktop Panini Strength | 0 | 0–1. Reduces edge stretching at wide FOVs (r.Upscale.Panini.D) |
| Desktop Eye Height | 0 | 0 = spawn height above the floor (160 cm if you spawn on the floor, e.g. the template's PlayerStart). Any other value sets a fixed eye height (cm) |
| Desktop Max FPS | 60 | Unreal frame cap without a headset (0 = no cap). Leaves CPU for the tracker |
| Show Status Text | on | On-screen hints (connection, calibration, lost hands) |
| Show Hand Debug | off | Prints hand direction and gesture data |

## Tracker options

Pass extra arguments to the `.bat` from a command prompt:

- Unreal runs on another PC: `Start_WebcamTracker.bat --osc-ip <that PC's IP>`
- Several cameras: `Start_WebcamTracker.bat --camera 1`
- Point gesture debug panel: `--point-debug` (toggle with **D** in the window)
- Press **q** in the preview window to quit.

## Troubleshooting

- **Hands never appear.** Check that the preview window shows your hands and
  the FPS counter is running. Check that the level has a
  `BP_WebcamOSCReceiver` and that the OSC plugin is enabled. If Unreal runs on
  another PC, check `--osc-ip`, and allow UDP 7000 through the firewall.
- **Jitter.** Light the room better, or lower the smoothing values.
- **Tracking feels slow.** If the tracker FPS is below ~25, lower Desktop Max FPS (e.g. 45).
- **A hand rotates the wrong way.** Turn on Show Hand Debug. In the
  calibration pose, `finger ≈ (0,0,1)` and `palm ≈ (1,0,0)`.

## OSC message format (for developers)

`/hand/{left|right}/landmarks` is a float array:

| Index | Content |
|---|---|
| 0–62 | 21 landmarks × (x, y, z), normalized image coordinates |
| 63 | grab flag (1 = fist) |
| 64–66 | finger direction (Unreal pawn axes) |
| 67–69 | palm normal (Unreal pawn axes) |
| 70 | trigger flag |
| 71 | hand distance to camera in metres (0 = unknown) |
| 72 | point flag (1 = teleport aim; right hand only) |

Other messages: `/hand/{side}/wrist`, `/hand/{side}/valid`, `/head/valid`,
`/head/matrix`, `/head/nose`.

## Support the project

WebcamXR is free and experimental. Bug reports and ideas are welcome on
GitHub: https://github.com/eknkayaa/WebcamXR/issues

If it saves you time, you can support development on Buy Me a Coffee: https://buymeacoffee.com/eknkayaa

## License

WebcamXR is released under the MIT License (see `LICENSE.txt`).

The tracker uses Google MediaPipe and its hand and face landmark models
(Apache License 2.0). See `NOTICE.txt` and `LICENSE-Apache-2.0.txt`.
