# Video Producer: setup notes

Setup facts that the administration page leaves out and that the admin reference does not hold.

## Install the Wan presets with their own recipe

For a preset that recommends files, PotionUI offers a recipe named `<preset> modes` behind the **Install** button it shows an admin next to a mode whose files are missing.

On PotionUI 0.2.0, the last step of that recipe, **Check each mode**, fails for **Wan 2.2 14B Lightning I2V (12 GB)** with "The 'video' setup for '…' didn't check out".

The steps before it have already downloaded and registered the files, set them as the preset's defaults, installed the preset and assigned it to you; the failure undoes none of that.

Instead, run the Video Producer recipe that has the preset's name in **Admin › Recipes**: it downloads the files Video Producer uses with that preset, installs it and assigns it to you, with no **Check each mode** step.

Like every recipe, it assigns the preset to the admin who runs it only. Assign it to your users or groups in **Admin › Presets**.
