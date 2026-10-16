# Video Producer: reference prototype next to the plugin

Each composite puts the reference interface prototype on the left and the plugin running inside PotionUI on the right. Both sides are 1440 x 900 captures of the French interface in the default project state (no selection unless the line says so), reduced to 50 % and joined into one 1440 x 450 JPEG (quality 80). Every screen exists in a dark and a light version: `<screen>-dark.jpg` and `<screen>-light.jpg`.

Full-size captures (1440 x 900 and 1280 x 720, plus an English series of the plugin) are not kept in the repository.

## Differences that apply to every composite

- The plugin page sits next to the 56 px PotionUI icon rail and the host page header is hidden. At 1440 px the plugin area is 1376 px wide (rail plus an 8 px scrollbar gutter reserved by the host), so columns are about 20 px narrower: storyboard thumbnails are 12 px shorter, and the layouts that wrap earlier are kept on one line.
- Status bar: a `FR` / `EN` language button replaces the prototype label, and the active tool reads "Outil Sélection (A)" instead of "Outil : Sélection (A)". The project bar reads "Enregistré à —" until the first save.
- Shortcut hints follow the platform (`Ctrl+G` instead of `⌘ G`) and decimals follow the locale (`6,0 s` instead of `6.0 s`).
- Hand-made controls are replaced by host components (Button, IconButton, Tooltip, Badge, SegmentedControl, Alert, Switch): labels are weight 500 instead of 400, secondary fills are one step lighter (`surface-3` instead of `surface-2`), badges are 22 px tall with wider padding where the prototype chips are 20 px, and settings toggles are 44 x 24 px switches instead of checkboxes.
- The engine selector shows the backend name given by the host ("Ce poste · RTX 4090") instead of "ComfyUI · RTX 4090".
- Vocabulary is unified: "Instruction locale" (prototype: "Prompt de plan", "Instructions locales du plan"), "Control video" (prototype: "Vidéo de contrôle", "contrôle"), "Sous-titres".
- Chips drawn over images (shot numbers, durations, scope tags, take letters, "Prise finale · IA simulée") use a dark scrim with white text in both themes, as in the prototype. Status labels on the take thumbnails (draft, final, to redo) are white instead of tinted.

## Screens

| Composite | What differs besides the list above |
|---|---|
| [media-dark](media-dark.jpg) / [media-light](media-light.jpg) | Same media browser, viewer and inspector. The viewer badge text is white instead of muted grey. |
| [director-direction-dark](director-direction-dark.jpg) / [light](director-direction-light.jpg) | The "Points à vérifier" box keeps its amber frame, but its title and links are set in the foreground colour (the prototype tints them), with a 12 px medium title instead of 11 px semibold; its first item wraps onto four lines instead of three. The summary reads "1 frame" where the prototype reads "2 frames": frames now belong to a shot, so the frame at the 6,0 s boundary counts for the next shot. |
| [director-references-dark](director-references-dark.jpg) / [light](director-references-light.jpg) | Same board; family descriptions use the muted text colour instead of the subtle one, and the control-video family reads "Control video et trajectoires de caméra". |
| [director-results-dark](director-results-dark.jpg) / [light](director-results-light.jpg) | Same takes list; "Utiliser" and "Assembler les prises choisies" are host buttons. |
| [storyboard-dark](storyboard-dark.jpg) / [light](storyboard-light.jpg) | Three columns as in the prototype, with narrower cards; the synopsis field is 4 px taller; "Plan" and "Proposer un découpage" are host buttons. |
| [scene-dark](scene-dark.jpg) / [light](scene-light.jpg) | Practically identical. Mode pills are host badges (22 px instead of 20 px) and the disabled "Suivre la trajectoire" label is lighter. |
| [mixer-dark](mixer-dark.jpg) / [light](mixer-light.jpg) | Active insert chips use a neutral fill with a signal border instead of a blue-tinted fill; muted labels are one step lighter; faders are 2 px shorter; "Auto off" is 11 px instead of 10 px; the viewer title card is never smaller than 10 px (5,5 px in the prototype). |
| [inspector-shot-dark](inspector-shot-dark.jpg) / [light](inspector-shot-light.jpg) | Shot 1 selected, "Plan" tab. Same fields; "Voir les prises" and the flag buttons use weight 500 labels and a lighter fill. |
| [inspector-takes-dark](inspector-takes-dark.jpg) / [light](inspector-takes-light.jpg) | Shot 1 selected, "Prises" tab. "2 brouillons de plus" is a host button. |
| [inspector-color-dark](inspector-color-dark.jpg) / [light](inspector-color-light.jpg) | Shot 1 selected, "Couleur" tab. Same presets and sliders. |
| [prefs-dark](prefs-dark.jpg) / [light](prefs-light.jpg) | No theme selector (the theme follows the host); a language selector is added, and the magnetism, guides and "Raccourcis à une touche" options are host switches. |
| [shortcuts-dark](shortcuts-dark.jpg) / [light](shortcuts-light.jpg) | Same list in two columns with platform key names (`Ctrl+Alt+K`, `Retour arrière`). |
| [about-dark](about-dark.jpg) / [light](about-light.jpg) | The subtitle reads "Plugin PotionUI · version 0.1.0" instead of "prototype d'interface 0.2". |
| [simulations-dark](simulations-dark.jpg) / [light](simulations-light.jpg) | Titled "Fonctions simulées" instead of "Simulations du prototype"; nine entries instead of eleven (the "Erreurs de génération" and "Assistant PotionUI" entries are gone), and "Control video" replaces "Vidéo de contrôle". |
| [index-dark](index-dark.jpg) / [light](index-light.jpg) | Same timeline index; decimal commas. |
| [tasks-dark](tasks-dark.jpg) / [light](tasks-light.jpg) | Identical. |
| [tour-dark](tour-dark.jpg) / [light](tour-light.jpg) | Identical. |
| [export-dark](export-dark.jpg) / [light](export-light.jpg) | The warning list keeps its amber frame with foreground-coloured text; the "Stems audio par rôle" option is a host switch; "Simulé" badges sit next to the format and stems controls and an "Export vidéo simulé" note sits in the footer, where the prototype labels the button "Exporter · simulé". |
| [advanced-dark](advanced-dark.jpg) / [light](advanced-light.jpg) | The current step is marked by a signal border and an outlined number instead of a filled blue number on a grey fill; the counters read "1 instruction locale" and "1 frame de référence" (the prototype says "2 frames"); the empty motion-guide state is a single row with its "Préparer une caméra en Scène 3D" button on the right. |
| [library-dark](library-dark.jpg) / [light](library-light.jpg) | Tag rows are 22 px instead of 20 px (cards 84 px instead of 82 px), the variable fields are 28 px instead of 24 px, the preview field is shorter, "Compléter le texte" / "Remplacer" are host segmented buttons. |
| [sound-dark](sound-dark.jpg) / [light](sound-light.jpg) | The brief field is 13 px instead of 12 px; the empty result is a host empty state with a semibold title. |
| [prompt-editor-dark](prompt-editor-dark.jpg) / [light](prompt-editor-light.jpg) | Titled "Éditer l'instruction locale" with the field "Texte de l'instruction locale" (prototype: "Éditer le prompt", "Instruction du plan"); the field has the focus on opening (signal border); "Bibliothèque & assistant" is a 24 px button instead of a 16 px one; the range summary reads "1 frame · 4 références" instead of "2 frame(s) · 4 référence(s)". |

## Starter project

The `starter-<screen>-<theme>.jpg` images show the plugin alone, with no prototype next to them: the prototype opens on a different sample project. They are 720 x 450 reductions (50 %, quality 80) of 1440 x 900 captures of the French interface in the dark and light themes.

They show the starter project "Aube sur le port" that the plugin opens on: five shots to generate, five prompts, a FL2VA profile, and nothing else (no media, no reference, no 3D scene, no take, no frame). The Production bar therefore warns that the range holds no frame, and the project reads "Non enregistré" until it is saved.

| Image | What it shows |
|---|---|
| [starter-media-dark](starter-media-dark.jpg) / [light](starter-media-light.jpg) | Empty media browser with its import action; the viewer shows the project title card and a "Générer des brouillons" action with its estimate. |
| [starter-director-direction-dark](starter-director-direction-dark.jpg) / [light](starter-director-direction-light.jpg) | "Direction" tab: the generation summary (global direction, local instruction, "0 référence · 0 frame · 0 Control video") and the H3 profile choice. |
| [starter-director-references-dark](starter-director-references-dark.jpg) / [light](starter-director-references-light.jpg) | "Références" tab and board: no reference yet, with the "+ Référence" action. |
| [starter-director-results-dark](starter-director-results-dark.jpg) / [light](starter-director-results-light.jpg) | "Résultats" tab: no take yet. |
| [starter-storyboard-dark](starter-storyboard-dark.jpg) / [light](starter-storyboard-light.jpg) | Five "Plan à générer" cards, the synopsis, the "À produire" list and the stage counters (five shots to generate). |
| [starter-scene-dark](starter-scene-dark.jpg) / [light](starter-scene-light.jpg) | Empty 3D scene space with the "Scène d'essai" and Gaussian Splat actions. |
| [starter-mixer-dark](starter-mixer-dark.jpg) / [light](starter-mixer-light.jpg) | Console with the default dialogue, music, ambience and sound-effects strips and the buses; no sound in the project. |
| [starter-storyboard-generated-dark](starter-storyboard-generated-dark.jpg) / [light](starter-storyboard-generated-light.jpg) | After "Générer la sélection" in draft quality and the simulated generation (about four seconds): shot 1 has two draft takes whose thumbnails are drawn procedurally (a numbered gradient marked "Simulé"), the project reads "Modifié", and the Production bar reports "2 prises prêtes". |
| [starter-inspector-takes-dark](starter-inspector-takes-dark.jpg) / [light](starter-inspector-takes-light.jpg) | The same state with shot 1 selected and the inspector on the "Prises" tab: takes A and B, with the "Passer en final" action and the estimate for two more drafts. |

## Project persistence

`open-window.png`, `version-history.png` and `media-import.png` show the plugin alone, with no prototype next to them: the prototype has no saved projects. Each image joins the dark theme (left) and the light theme (right) of the English interface. They are 1:1 crops of 1440 x 900 captures (no reduction), kept as PNG: 1440 x 450 for the Open window, 1440 x 540 for the version history and 720 x 450 for the media panel.

The project is "Dawn at the harbour" with two other saved projects ("Alpine trailer", "Sketchbook"); modification times are spaced out so that the relative times and the version dates read differently.

| Image | What it shows |
|---|---|
| [open-window](open-window.png) | File > Open…: the saved projects, most recently modified first, each with its name, relative modification time, shot count and running time. A "Current" badge marks the open project, which has no delete button. "Alpine trailer" is selected (signal frame), so "Open" is enabled; the trash icon asks for a confirmation from the host before deleting; "New project" creates an empty project. |
| [version-history](version-history.png) | File > Version history…: the seven versions of the open project, latest first, each with its date and time to the second, shot count and optional label. "Current" marks the version in the editor, version 6 reads "Restored from version 3", and version 4 is selected, so "Restore" is enabled. |
| [media-import](media-import.png) | The Media panel after File > Import files…, which opened the host library picker: a video and an image are listed with their thumbnail, the real duration ("4.0 s") and the "Library" origin badge. |
