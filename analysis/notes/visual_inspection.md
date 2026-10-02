# Visual inspection: data_small train (20) vs val (22)

Observations only, no conclusions. Written 2026-10-02 after looking at every image in `data_small/train/images`
and `data_small/val/images`, plus the six error grids in `figures/b1_tile1024/errors/crops_*.png`.

**How I looked.** For each image I made a sheet with two parts:
- the whole image downsampled to 900 px on its long side;
- a 450 x 450 px crop at native resolution, shown at 2x with nearest-neighbour scaling. The crop is centred on the
  median GT box centre, with the GT boxes drawn.

The sheets were scratch files and are not in the repo. Each crop is a single region, so it can miss what the rest of
the image looks like. All pixel sizes below are eyeballed from these crops, not measured; section (d) of
`analysis/domain_shift.py` gives measured box sizes.

These are 42 of the 465 images (443 train + 22 val), so any train-side impression may not hold for the full train set.

## Colour / tone / exposure
- **Both splits have a warm, brown-orange cast and are dark overall.** This is common in both splits, so the colour
  cast alone does not separate them.
  - Train: 130, 1478, 1707, 2359, 2416, 604, 871.
  - Val: 1530, 1929, 2292, 2470, 2472.
- **Very dark / underexposed images.** Train: 1610, 1945, 2020, 2150. Val: 1399, 1447, 1456, 1929.
  - 1399 and 1447 (val) are the darkest I saw. Detail is barely visible without boosting the brightness.
- **Haze and low contrast (washed-out brown, objects faint).**
  - Train: 2459, 610.
  - Val: 2460, 2470, 2472, also 1206 and 1211 to a lesser degree.
  - In 2470 and 2472 (val) trucks are hard for me to see at all in the native crops.
- **Saturated, crisp colour with clean daylight.**
  - Train: 2064, 2359, 2591, 73.
  - Val: 1362, 1457, 20, 2139, 31, 1530.
  - 2139 (val) has a very saturated dark-blue sea and hard shadows.
- **Clouds.**
  - Train: 1820, 1945, 2020, 2150 (mostly cloud), 2459, 610, 903.
  - Val: 1181, 1206, 2139, 2308, 2460, 2470, 2472.

## Sharpness / blur / resampling
- **Strong blur, a smooth "upsampled" look with no fine edges, vehicles as soft blobs.**
  - Val: 2292, 2308, 2543. In these native crops, trucks and cars are blobs without structure.
  - Train: 2020 (also hazy), 1945 (blocky, smeared). These are the closest train examples I saw, and I find them
    less extreme than val 2543 and 2292. **Unsure:** this is a visual impression; (a) measures it.
- **Sharp images with clear vehicle structure** (cab, trailer and roof edges visible).
  - Train: 2359, 2064, 871, 2591.
  - Val: 1457, 20, 31, 2391, 2293.
- **Ringing / over-sharpening halos.** Train 73 (around buildings). Val 1530 (pink-white blob artefact on a roof).

## Compression and other artefacts
- **Multicoloured speckle on water**: bright red/green/blue single-pixel sparkles.
  - Strong in val 1456 and 1447, also visible on water in val 1399.
  - I did not see this in the train water images (1707, 1478, 2150). **Unsure** whether it is sensor noise,
    pansharpening, or glint.
- **No-data border**: train 2591 has a black strip down its left side. I saw none in val.
- I saw no obvious JPEG block artefacts. The files are PNG, but the source imagery may have been compressed earlier;
  **unsure**.

## Apparent ground resolution (eyeballed)
- **Typical case:** in most images of both splits a car is roughly 8-12 px long and a typical truck roughly 20-40 px
  long. Train: 2064, 2591, 903, 2371. Val: 1211, 1457, 2384, 31.
- **Val 2391 (7072 x 5932 px, the largest image) looks finer-resolution.**
  - Trucks look roughly 2-3x longer in pixels than in most other images (box trailers ~40 x 60+ px).
  - It is also visibly soft, so it may have been upsampled. **Unsure** which.
  - It contributes many missed Box trucks in `crops_missed.png`.
- **Val 2460 is small (1596 x 1598 px).** The vehicles in it look about as small as elsewhere, so it may simply cover
  a smaller area. **Unsure.**
- **Val 2292 and 2543 look coarser** (vehicles as few-pixel blobs), consistent with the blur noted above.

## Viewing angle / shadows
- **Off-nadir lean** (tall buildings and stadiums show sides; long shadows).
  - Train: 2591, 1945.
  - Val: 1457 (tower shadows), 2139, 2308.
- **Mostly near-nadir:** the rest of both splits.
- **Long, hard shadows:** val 1457, 2139, 20; train 2359, 73.
- I did not see a systematic angle difference between the splits.

## Scene types
- **Train sample:** many different scenes.
  - Airports: 1086, 130, 1478, 2064, 2371.
  - City / residential: 1610, 2020, 2591, 1820.
  - Logistics park with semi-trailers: 2359.
  - Highway: 871, 2150, 903.
  - Desert mining: 604. Arid town: 610. Port with haze: 2459. Waterfront construction: 1707. Industrial: 73.
- **Val:** more ports and dense truck yards.
  - Ports: 1399, 1447, 1456, 20, 2293, 2460.
  - Industrial areas with truck yards: 1181, 1206, 1211, 2470, 2472, 2384, 1929.
  - Airports: 1362, 2391, 2139. Gulf-style city: 1457. Dense city: 2308, 1530. Arid port town: 31.
- **Regions (my guess, uncertain):**
  - Several val images look like the same West-African-style city: 1181, 1206, 1211, 2470, 2472, 2460. They share
    the hazy brown tone and corrugated-roof industry.
  - Train 2459 looks similar in tone and haze.

## Season / vegetation
- **Green vegetation:** val 1362, 1929, 2139, 2384; train 2020, 1945, 903, 1820.
- **Arid / dry:** val 20, 31, 1457, 2292; train 604, 610, 2416, 130, 2359.
- I see no clear seasonal split between train and val. **Unsure**, since season is hard to judge from overhead.

## How trucks look, per class (from the crops with drawn boxes)
- **Cargo Truck (red):**
  - Mostly small rigid trucks or vans, ~20-40 px. Train: 2371, 903, 2591. Val: 1211, 1457, 2139, 1929.
  - In 1929 (val) the Cargo boxes in the dense yard overlap heavily and cover a cluster of vehicles. I could not tell
    whether each box sits on one vehicle.
- **Truck w/Box (green):**
  - Box trucks and semi-trailers.
  - Train 2359 and 871: clean white trailers, ~45-60 px.
  - Val 20 and 2384: smaller, darker trucks, ~25-40 px.
  - Val 1181: the Box boxes are large (~60-100 px) and drawn around yellow/orange trailers in a dense yard.
  - Val 2391: large white trailers, many missed.
- **Truck w/Flatbed (blue):**
  - Train 130, 1820, 2416: a few examples. In 130 two overlapping blue boxes cover what looks like one object.
  - Val 1447 has 35 Flatbed boxes on dark trucks along a quay; I find them hard to see.
  - In val 1456 the Flatbed boxes sit on a row of near-identical white vehicles parked among dozens of identical
    unlabelled ones. **Unsure** what distinguishes the labelled ones.
- **Truck Tractor (yellow):**
  - Train 2359: a tractor with a trailer.
  - Val 31: small bare cabs, ~20-25 px.
  - Val 1181: a dense cluster of tractor boxes overlapping each other.
- **Truck w/Liquid (magenta):** tankers. Train 2150: one on a highway, ~45 px. Train 610: a small one. Few examples
  overall.

## Labelling observations (both splits)
- **Visible but unlabelled trucks.** I think I see trucks without boxes, but cannot be sure they belong to the five
  classes.
  - Train: 1610 (a yard with many trucks, only some boxed), 2359 (one white semi unboxed), 903.
  - Val: 1181 (many yellow trailers), 1456 (identical white vehicles), 2293.
- **Some val background-FP crops look like trucks to me.**
  - In `crops_bkg.png`: 31.png conf 0.56 and 0.25, 1211.png conf 0.50 and 0.55, 2308.png conf 0.28.
  - **Unsure** whether they are unlabelled trucks or other vehicles.
- **Some GT boxes look offset from the vehicle they seem to describe.**
  - In `crops_loc.png` and `crops_both.png` (2470 and 2472), the orange GT box is shifted by about half a box from
    the dark object that the red prediction sits on.
  - **Unsure:** in these hazy images I cannot see the truck well enough to say which box is right.

## Error-grid observations (`figures/b1_tile1024/errors/`, threshold set conf 0.25)
- **`crops_missed.png`:**
  - Missed trucks come from 2391 (large, clearly visible white trailers), 2543 (blurred blobs), 2460 and 2472 (haze),
    and 1399 (very dark).
  - The missed Liquid examples in 1362 are clearly visible tankers.
- **`crops_bkg.png`:**
  - Background FPs cluster in 1206, 1211, 2470, 2472, 2293 and 31.
  - No Truck w/Liquid row: none were sampled at this threshold.
- **`crops_cls.png`:** confusions mostly Box ↔ Cargo, Tractor ↔ Box and Flatbed ↔ Cargo. Many come from the hazy
  images 2470 and 2472.
- **`crops_dupe.png` is empty:** no duplicate errors at conf 0.25.

## Montage observations (`figures/domain_shift/montage_*.png`, 160 px native windows, same scale)
Sampling: seeded, at most 2 crops per image.
- **Val 2308, 2391 and 2543 look larger and softer.** The same-scale windows show vehicles noticeably larger
  (Liquid in 2308: boxes 104 x 38 and 104 x 68 px; cars in 2391 ~20 px long) and softer than in most train windows.
  - In the train windows, cars are typically ~10 px long (2064, 2371, 903).
  - **Unsure** whether this is a finer ground sample distance or upsampling. These images are soft rather than
    detailed, so upsampling is possible.
- **Train 2459 Box/Liquid windows:**
  - Dense, hazy, low-contrast rows of trailers. The labelled objects are faint rectangles.
  - The closest-looking val windows are 2460, 2470 and 2472, all from hazy industrial / port scenes.
- **Truck w/Liquid: the val windows I see are from different scenes than the train windows.**
  - Val (green-vegetation depot in 1362, dark port in 1399, city in 2308) vs train (mostly 2459's hazy trailer rows,
    plus 2150, 610, 73).
  - The full-dataset median sqrt(area) for Liquid is 41.5 px in val vs 25.5 px in train
    (`figures/domain_shift/tables/box_sizes_full_dataset.csv`), from 20 val boxes.
