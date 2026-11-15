# Video Producer

Video Producer is a PotionUI plugin for making short films, in one page, with AI video: a user writes a synopsis, splits it into shots, generates draft takes, keeps the best ones and renders them in final quality, then cuts, matches the colors, mixes the sound and exports the film. Every user can use it once an admin has set it up. Install it from this catalogue:
```text
https://raw.githubusercontent.com/M4-S4/potionui-plugins/main/plugins.json
```

It needs PotionUI 0.2.0 or later. This page is for the administrator who installs and sets it up; users read **Video Producer: user guide**.

1. In **Admin › Plugins**, click **Install from catalog**, paste the catalogue URL and click **Read catalog**.
2. Click **Install** next to **Video Producer**, then **Install** in the window that asks to confirm. It arrives in the list, disabled.
3. Turn it on with its switch. A **Video Producer** entry appears in the sidebar for every user, and the plugin shows **Setup needed** until the setup steps of its **Overview** tab are done.

To install by hand instead, unzip the release archive into `content/plugins/local/`, then click **Scan for Plugins** in **Admin › Plugins** and enable the plugin.

![The Storyboard of the starter project "Lumi’s potion" in the dark theme, after its first takes: four shots on the band, their takes, the monitor and the production deck](https://raw.githubusercontent.com/M4-S4/potionui-plugins/main/video-producer/docs/screenshots/starter-storyboard-generated-dark.webp)

## What it needs on the server

- **PotionUI 0.2.0 or later**, from the official repository, with an admin account.
- **A video engine**: the native engine of this machine, a Studio server (a PotionUI Remote Native worker), or a cloud backend with a video model turned on in its catalog, in **Admin › Backends**.
- **At least one Video Director preset**, installed and assigned in **Admin › Presets**, with its model files downloaded in **Admin › Downloads** (see Choose the presets); a MiniMax-H3 preset needs its five model files.
- **For final takes**, SeedVR2 or RTX Upscale (from the NVIDIA RTX Upscale plugin), assigned to the users; assign Video Tools or RTX Upscale too, for the frame rate.
- **A writable work folder** in the file storage directory of the server (see The work folder).
- **ffmpeg and ffprobe** on the PotionUI server, to export videos and to start a shot from the last image of the previous one (see ffmpeg).
- **For the assistant**, optionally, a language model that can call tools, assigned to the users.
- **For a local engine**, an NVIDIA card of the Turing generation or newer, with driver 580 or newer. Video needs more than PotionUI's minimum: plan on the card and the system RAM below.

The **Overview** tab of the plugin, in **Admin › Plugins**, checks these as six setup steps: each one reads the instance as it is now, says what is missing and, when it can, offers a button to the right admin page. A preset installed but assigned to no one does not count: an administrator sees every installed preset, but other users only those assigned to them or to their groups. That is why something can work for you and not for a user.

## Choose the presets

What your users can make depends on the graphics card. Video Producer ships six presets set up for 8 GB and 12 GB cards, each with a recipe in **Admin › Recipes** that downloads its files, installs it and assigns it.

| Card | Presets to install and assign | System RAM | What the users get |
|---|---|---|---|
| 8 GB | **Wan 2.1 T2V 1.3B (8 GB)**, **Wan 2.2 TI2V 5B (8 GB)** | 32 GB | 480p clips of up to 5 seconds; the 5B can start from a first frame. |
| 12 GB | **MiniMax-H3 (12 GB)**, **MiniMax-H3-Fast (12 GB)** | 64 GB | Video with sound, up to 15 seconds, at 832×480 by default; **MiniMax-H3 (12 GB)** also follows references (Ref2VA). |
| 12 GB | **Wan 2.2 14B Lightning (12 GB)** with its companion **Wan 2.2 14B Lightning I2V (12 GB)** | 64 GB | Fast text to video, and image to video for the shots that start from an image. |

The other Video Director presets of PotionUI (Wan, LTX, MiniMax-H3 and its variants, OpenRouter Video, Gemini Video) work too once assigned. A preset that does not fit the card streams part of the model through the system RAM: the generation still runs, more slowly, and the production deck warns "Will stream through host RAM, slower".

### What to expect on a 12 GB card

Measured on an NVIDIA RTX 4070 Ti with 12 GB, with the native engine of PotionUI. Tell your users, and set **Max generations in flight per user** and the monthly GPU minutes with these times in mind.

| Engine | A 5-second take at 832×480 |
|---|---|
| **MiniMax-H3 (12 GB)**, 20 steps | about 7.5 min |
| **MiniMax-H3-Fast (12 GB)**, 8 steps | about 6 min |
| **Wan 2.2 14B Lightning (12 GB)**, 6 steps | about 3.5 min, once its 38 GB of files are read |
| **Wan 2.2 14B Lightning I2V (12 GB)**, 4 steps | 2 min 15 s to 2 min 45 s |
| **Wan 2.1 T2V 1.3B (8 GB)** and **Wan 2.2 TI2V 5B (8 GB)**, card limited to 8 GB | about 5.5 and 4.5 min |

The core Wan 2.1 / 2.2 14B preset in 30 steps takes about 26 minutes, and a SeedVR2 upscale of a 5-second draft to 1080p about 2.5 hours. What runs out of memory on this card, the RAM each preset used and the other measures are in the admin reference (Hardware, Hardware tiers, LTX) and in the Bench measures of the technical reference.

## Getting started

### The work folder

Video Producer writes its work files in `<file storage directory>/video_producer/`: the exports, the frames taken from the shots, and the sound cut for the generation. This folder must be local and writable on the server, even when PotionUI keeps the media in object storage. If the setup step says "Video Producer cannot write in …", set a writable file storage directory in **Admin › Settings**, or mount one into the container.

### ffmpeg

Video Producer uses `ffmpeg` and `ffprobe`, as found on the `PATH` of the PotionUI process, to export the videos, to cut the sound sent with a shot, to start a shot from the last image of the previous one, and to read the frame rate of a take. Install both so that the PotionUI process finds them, then restart PotionUI; in Docker, check that the image contains them. Without ffmpeg, users cannot export a video (the export window says so) and those shots are blocked; the files for other software and the subtitles still work. The admin reference lists the encoders and filters the export uses.

![The Inspector in the dark theme, on the Takes tab of a shot: a draft take made by MiniMax H3 (12 GB), labelled Selected](https://raw.githubusercontent.com/M4-S4/potionui-plugins/main/video-producer/docs/screenshots/starter-inspector-takes-dark.webp)

### Then

- **Settings**, in **Admin › Plugins › Video Producer › Settings**: **Draft resolution** (832×480 by default), **Max generations in flight per user** (2), **Export cache per user (MB)** (2048) and **Versions kept per project** (50). The language of the interface is each user's choice, and every prompt is in English.
- **Quotas**: a plan of **Admin › Users › Plans** can limit **Video Producer GPU minutes per month**. Cloud models use no GPU minutes: their spending falls under **Cloud spend per month**.
- **Engines**: ComfyUI backends are not offered, and only administrators see the cost estimate of a cloud take.
- **Finals**: SeedVR2 upscales a draft to 720p, 1080p, 1440p or 4K and restores detail, slowly; RTX Upscale does it in seconds without adding detail and needs the `nvidia-vfx` package; the guided render needs LTX-2.5 and its Layout-To-Render IC-LoRA. Upscalers, interpolators and the guided render run on a native backend.
- **The assistant**, a mode of the PotionUI chat named **Video Producer assistant**: in **Admin › LLM / Assistant**, click **Add configuration**, add a model that can call tools and assign it to the users. When the model shares the graphics card with the video, use a local llama.cpp `llama-server`, added as an OpenAI configuration.

The admin reference gives every detail: the full texts of the setup steps, the bounds of each setting, how minutes are reserved and refunded, how a backend is chosen, the tested assistant setups.

### Documentation

Every page is in **Admin › Documentation** once the plugin is enabled; the user guide is also in the Documentation page that every signed-in user sees.

- **Video Producer: administration**, this page, for the administrator.
- **Video Producer: admin reference** ([docs/admin-reference.md](docs/admin-reference.md)): the detail behind this page, the measures, the licenses and the troubleshooting list.
- **Video Producer: user guide** ([docs/user-guide.md](docs/user-guide.md)): for every user, how to make a film.
- **Video Producer: complete reference** ([docs/user-reference.md](docs/user-reference.md)): every control and message explained.
- **Video Producer: technical reference** ([docs/technical-reference.md](docs/technical-reference.md)): for developers, how the plugin works, its routes, tables, errors, tests and bench measures.

![Director AI in the dark theme, Direction view: the five fields of the global direction, and the prompt the first shot will send to the engine](https://raw.githubusercontent.com/M4-S4/potionui-plugins/main/video-producer/docs/screenshots/starter-director-direction-dark.webp)

## Limits to know

- **MiniMax-H3 license.** Its weights are under the MiniMax H3 Community License, not an open-source license, whose territory excludes the European Union, the United Kingdom, the United States and the Republic of Korea, what the model generates included. Read the MiniMax-H3 reminder of the admin reference before you enable it.
- **LTX-2.5 license.** LTX-2.5 and its IC-LoRAs are under the LTX-2.x Community License of Lightricks; read it, and the summary in the admin reference, before you install LTX-2.5.
- **MiniMax-H3 files.** The Video mode needs the `fl2va` DiT file and the References mode the `ref2va` one; a wrong pick runs without an error and ignores the references.
- **Validation.** PotionUI's own documentation still lists the MiniMax-H3 family as not validated on real weights; the 12 GB presets of this plugin were measured on a 12 GB card.
- **One export at a time.** The server runs one video export at a time, for everybody: the others stay Queued.
- **The Install button of a Wan preset.** For a preset that recommends files, PotionUI offers a recipe named `<preset> modes` behind the **Install** button it shows an admin next to a mode whose files are missing. Its last step, **Check each mode**, fails on PotionUI 0.2.0 for Wan 2.2 14B Lightning I2V (12 GB), and on PotionUI 0.1.0 for the four Wan presets, with "The 'video' setup for '…' didn't check out". The steps before it have already downloaded and registered the files, set them as the preset's defaults, and installed and assigned the preset; the failure undoes none of that. Instead, run the Video Producer recipe that has the preset's name in **Admin › Recipes**: it downloads the files Video Producer uses with that preset and installs and assigns it, with no **Check each mode** step.

Every other message, with its cause and its fix, is in the troubleshooting list of the admin reference.

## Changes

### What changed in 0.6.0

- **From a kit**: wherever Video Producer imports or chooses a media, a **From a kit** button opens your PotionUI kits; each item taken becomes an ordinary media of the project, which no longer follows the kit.
- The two MiniMax-H3 (12 GB) presets follow the prompt syntax, forms and video pipeline of the MiniMax-H3 presets of PotionUI 0.2.0, with no change of default or render; the **Edit Engine** window of **MiniMax-H3 (12 GB)** gains the **Colour hold** slider, at 0.
- The generation summary window, which nothing opened any more, is gone: what it showed is in the Review and References views of Director AI.
- Needs PotionUI 0.2.0 or later.

### What changed in 0.5.2

- When the assistant or another window changes a project with cross-dissolves while you edit it, the merged project keeps its dissolves playing, on the page and in the export, and every prompt and keyframe stays in its place.

### What changed in 0.5.1

- In a project without a cross-dissolve, the export no longer puts a black frame after a shot whose length is off the 0.04 s grid, and every cut falls on the frame of its time, so the picture and the sound of the shots stay in step with the music and the subtitles.
- A shot split with the blade plays on in the export with no repeated or skipped frame; the EDL, OpenTimelineIO and FCPXML files put every cut on the frame of the video.
- The export of some projects changes, and the first edit of a project made with an earlier version can move a shot 0.04 s later.

### What changed in 0.5.0

- **Cross-dissolve** between two shots, of 0.4, 0.6, 1, 1.2 or 2 seconds, chosen under **Transition** in the Inspector; the viewer, the export and the exchange files hold it, and the film gets shorter by its length.
- A project that carries a cross-dissolve does not open with Video Producer 0.4.0 or older.

### What changed in 0.4.0

- **Finishing** at export: Fix colours, a LUT, a Look, a Vignette and Film grain on a web or social export, saved to the library as a new video; the master and the archive are never finished.
- **Write the prompt**: the language model of PotionUI proposes the prompt of a shot, and nothing changes until **Use this prompt**.
- **MiniMax-H3-Fast (12 GB)** continues the previous take with **Chain by › Continuation** in FL2VA, about 23 % slower than starting from the last frame; with MiniMax-H3 in Ref2VA, a shot can set its own **Card words beside a picture**.

### What changed in 0.3.2

- Only the 12 GB MiniMax-H3 presets of Video Producer continue an FL2VA shot chained by continuation; the MiniMax-H3 presets of PotionUI start it from the last frame again, instead of replaying the end of the previous shot.

### What changed in 0.3.1

- The MiniMax-H3 setup step names a missing model file instead of showing that it could not be checked.

### What changed in 0.3.0

- **Chain by continuation** on MiniMax-H3: an FL2VA shot on MiniMax-H3 (12 GB) continues the previous take, and a Ref2VA shot receives the end of the previous take, with a guard for the text encoder of a 12 GB card.
- **Import a cut as shots**: a video is split at its cuts, and each piece becomes the source or the motion reference of a shot.
- A project chooses what a MiniMax-H3 Ref2VA card says beside its picture: its words, its label, or the picture alone.
- The documentation becomes a short administration page, a user guide, and the complete, admin and technical references.

### What changed in 0.2.0

- First release in the catalogue, with **Lumi’s potion**, a starter project of four shots, each showing one feature, that opens for a new user.
- Every prompt is in English, and the Prompt language setting is gone.
- The page ships its own stylesheet, so it keeps its layout on a PotionUI whose frontend was built without the plugin.

## Update

Open the catalogue again: a newer version shows **Update to** and its version number in place of **Install**.

- Read the catalogue at the same URL as the first time. PotionUI updates a plugin only from the exact address it was installed from; under any other address, even one that serves the same file, the plugin shows **Installed elsewhere** with no update.
- After an update, restart PotionUI.

If PotionUI refuses an update with the message that the new version did not start, the version you had is back in place. Then:

1. disable the plugin;
2. restart PotionUI;
3. update it;
4. enable it.

## License

GPL-3.0, like PotionUI. The licenses of the models you use with it are separate: read those of MiniMax-H3 and LTX-2.5 in the admin reference. The three images of the starter project ship with the plugin under its license; the admin reference says how they were made.
