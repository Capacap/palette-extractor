#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10.0.0", "numpy>=1.24.0", "scipy>=1.11.0"]
# ///
"""
Unified color analysis pipeline.

Extracts color schemes from images and produces prose reports for LLM consumption.
Four stages: Data Preparation → Feature Extraction → Synthesis → Render
"""

import functools
import math
from html import escape as html_escape

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import label
from dataclasses import dataclass, field

# Type aliases for clarity
BinKey = tuple[int, int, int]
Position = tuple[int, int]


# =============================================================================
# Constants
# =============================================================================

JND = 2.3  # Just Noticeable Difference in LAB units
FINE_SCALE = 1.0  # Fine bins: 1 JND = 2.3 LAB units
COARSE_SCALE = 5.0  # Coarse bins: ~12 LAB units

# Image size limits (security: prevent decompression bombs)
MAX_IMAGE_PIXELS = 50_000_000  # 50 megapixels
MAX_IMAGE_DIMENSION = 10_000  # 10k pixels per side

# Algorithm parameters
HUE_CLUSTER_RANGE = 30  # Degrees within which hues are considered similar
MAX_GRADIENT_SEARCH_SEEDS = 50  # Limit gradient detection search
MAX_GRADIENT_CHAIN_LENGTH = 20  # Maximum colors in a gradient chain
MIN_SIGNIFICANCE_RATIO = 0.05  # Minimum significance relative to top color
MAX_NOTABLE_COLORS = 10  # Maximum colors to include in output
GRADIENT_ANGLE_THRESHOLD = 74  # Degrees; reject gradients with mean local angle above this

# Similarity penalty parameters (for diverse palette selection)
SIMILARITY_DELTA_E_THRESHOLD = 25.0  # LAB distance at which colors are "fully different"
COVERAGE_PROTECTION_ANCHOR = 0.05  # Coverage level (5%) that gets full protection from penalty
SIMILARITY_PENALTY_WEIGHT = 30.0  # Max penalty subtracted from significance score

# Family-based color selection parameters
FAMILY_CLUSTER_THRESHOLD = 30.0  # LAB distance for grouping colors into families
MIN_FAMILY_COVERAGE = 0.01  # Minimum coverage (1%) for a family to get automatic representation
ACCENT_CHROMA_THRESHOLD = 30  # Minimum chroma for low-coverage families to qualify as accents

# XKCD color survey data (949 colors)
# Source: https://xkcd.com/color/rgb/ (CC0 1.0 Public Domain)
# Format: (name, R, G, B)
XKCD_COLORS = (
    ('Cloudy Blue', 172, 194, 217),
    ('Dark Pastel Green', 86, 174, 87),
    ('Dust', 178, 153, 110),
    ('Electric Lime', 168, 255, 4),
    ('Fresh Green', 105, 216, 79),
    ('Light Eggplant', 137, 69, 133),
    ('Nasty Green', 112, 178, 63),
    ('Really Light Blue', 212, 255, 255),
    ('Tea', 101, 171, 124),
    ('Warm Purple', 149, 46, 143),
    ('Yellowish Tan', 252, 252, 129),
    ('Cement', 165, 163, 145),
    ('Dark Grass Green', 56, 128, 4),
    ('Dusty Teal', 76, 144, 133),
    ('Grey Teal', 94, 155, 138),
    ('Macaroni And Cheese', 239, 180, 53),
    ('Pinkish Tan', 217, 155, 130),
    ('Spruce', 10, 95, 56),
    ('Strong Blue', 12, 6, 247),
    ('Toxic Green', 97, 222, 42),
    ('Windows Blue', 55, 120, 191),
    ('Blue Blue', 34, 66, 199),
    ('Blue With A Hint Of Purple', 83, 60, 198),
    ('Booger', 155, 181, 60),
    ('Bright Sea Green', 5, 255, 166),
    ('Dark Green Blue', 31, 99, 87),
    ('Deep Turquoise', 1, 115, 116),
    ('Green Teal', 12, 181, 119),
    ('Strong Pink', 255, 7, 137),
    ('Bland', 175, 168, 139),
    ('Deep Aqua', 8, 120, 127),
    ('Lavender Pink', 221, 133, 215),
    ('Light Moss Green', 166, 200, 117),
    ('Light Seafoam Green', 167, 255, 181),
    ('Olive Yellow', 194, 183, 9),
    ('Pig Pink', 231, 142, 165),
    ('Deep Lilac', 150, 110, 189),
    ('Desert', 204, 173, 96),
    ('Dusty Lavender', 172, 134, 168),
    ('Purpley Grey', 148, 126, 148),
    ('Purply', 152, 63, 178),
    ('Candy Pink', 255, 99, 233),
    ('Light Pastel Green', 178, 251, 165),
    ('Boring Green', 99, 179, 101),
    ('Kiwi Green', 142, 229, 63),
    ('Light Grey Green', 183, 225, 161),
    ('Orange Pink', 255, 111, 82),
    ('Tea Green', 189, 248, 163),
    ('Very Light Brown', 211, 182, 131),
    ('Egg Shell', 255, 252, 196),
    ('Eggplant Purple', 67, 5, 65),
    ('Powder Pink', 255, 178, 208),
    ('Reddish Grey', 153, 117, 112),
    ('Baby Shit Brown', 173, 144, 13),
    ('Liliac', 196, 142, 253),
    ('Stormy Blue', 80, 123, 156),
    ('Ugly Brown', 125, 113, 3),
    ('Custard', 255, 253, 120),
    ('Darkish Pink', 218, 70, 125),
    ('Deep Brown', 65, 2, 0),
    ('Greenish Beige', 201, 209, 121),
    ('Manilla', 255, 250, 134),
    ('Off Blue', 86, 132, 174),
    ('Battleship Grey', 107, 124, 133),
    ('Browny Green', 111, 108, 10),
    ('Bruise', 126, 64, 113),
    ('Kelley Green', 0, 147, 55),
    ('Sickly Yellow', 208, 228, 41),
    ('Sunny Yellow', 255, 249, 23),
    ('Azul', 29, 93, 236),
    ('Darkgreen', 5, 73, 7),
    ('Green/Yellow', 181, 206, 8),
    ('Lichen', 143, 182, 123),
    ('Light Light Green', 200, 255, 176),
    ('Pale Gold', 253, 222, 108),
    ('Sun Yellow', 255, 223, 34),
    ('Tan Green', 169, 190, 112),
    ('Burple', 104, 50, 227),
    ('Butterscotch', 253, 177, 71),
    ('Toupe', 199, 172, 125),
    ('Dark Cream', 255, 243, 154),
    ('Indian Red', 133, 14, 4),
    ('Light Lavendar', 239, 192, 254),
    ('Poison Green', 64, 253, 20),
    ('Baby Puke Green', 182, 196, 6),
    ('Bright Yellow Green', 157, 255, 0),
    ('Charcoal Grey', 60, 65, 66),
    ('Squash', 242, 171, 21),
    ('Cinnamon', 172, 79, 6),
    ('Light Pea Green', 196, 254, 130),
    ('Radioactive Green', 44, 250, 31),
    ('Raw Sienna', 154, 98, 0),
    ('Baby Purple', 202, 155, 247),
    ('Cocoa', 135, 95, 66),
    ('Light Royal Blue', 58, 46, 254),
    ('Orangeish', 253, 141, 73),
    ('Rust Brown', 139, 49, 3),
    ('Sand Brown', 203, 165, 96),
    ('Swamp', 105, 131, 57),
    ('Tealish Green', 12, 220, 115),
    ('Burnt Siena', 183, 82, 3),
    ('Camo', 127, 143, 78),
    ('Dusk Blue', 38, 83, 141),
    ('Fern', 99, 169, 80),
    ('Old Rose', 200, 127, 137),
    ('Pale Light Green', 177, 252, 153),
    ('Peachy Pink', 255, 154, 138),
    ('Rosy Pink', 246, 104, 142),
    ('Light Bluish Green', 118, 253, 168),
    ('Light Bright Green', 83, 254, 92),
    ('Light Neon Green', 78, 253, 84),
    ('Light Seafoam', 160, 254, 191),
    ('Tiffany Blue', 123, 242, 218),
    ('Washed Out Green', 188, 245, 166),
    ('Browny Orange', 202, 107, 2),
    ('Nice Blue', 16, 122, 176),
    ('Sapphire', 33, 56, 171),
    ('Greyish Teal', 113, 159, 145),
    ('Orangey Yellow', 253, 185, 21),
    ('Parchment', 254, 252, 175),
    ('Straw', 252, 246, 121),
    ('Very Dark Brown', 29, 2, 0),
    ('Terracota', 203, 104, 67),
    ('Ugly Blue', 49, 102, 138),
    ('Clear Blue', 36, 122, 253),
    ('Creme', 255, 255, 182),
    ('Foam Green', 144, 253, 169),
    ('Grey/Green', 134, 161, 125),
    ('Light Gold', 253, 220, 92),
    ('Seafoam Blue', 120, 209, 182),
    ('Topaz', 19, 187, 175),
    ('Violet Pink', 251, 95, 252),
    ('Wintergreen', 32, 249, 134),
    ('Yellow Tan', 255, 227, 110),
    ('Dark Fuchsia', 157, 7, 89),
    ('Indigo Blue', 58, 24, 177),
    ('Light Yellowish Green', 194, 255, 137),
    ('Pale Magenta', 215, 103, 173),
    ('Rich Purple', 114, 0, 88),
    ('Sunflower Yellow', 255, 218, 3),
    ('Green/Blue', 1, 192, 141),
    ('Leather', 172, 116, 52),
    ('Racing Green', 1, 70, 0),
    ('Vivid Purple', 153, 0, 250),
    ('Dark Royal Blue', 2, 6, 111),
    ('Hazel', 142, 118, 24),
    ('Muted Pink', 209, 118, 143),
    ('Booger Green', 150, 180, 3),
    ('Canary', 253, 255, 99),
    ('Cool Grey', 149, 163, 166),
    ('Dark Taupe', 127, 104, 78),
    ('Darkish Purple', 117, 25, 115),
    ('True Green', 8, 148, 4),
    ('Coral Pink', 255, 97, 99),
    ('Dark Sage', 89, 133, 86),
    ('Dark Slate Blue', 33, 71, 97),
    ('Flat Blue', 60, 115, 168),
    ('Mushroom', 186, 158, 136),
    ('Rich Blue', 2, 27, 249),
    ('Dirty Purple', 115, 74, 101),
    ('Greenblue', 35, 196, 139),
    ('Icky Green', 143, 174, 34),
    ('Light Khaki', 230, 242, 162),
    ('Warm Blue', 75, 87, 219),
    ('Dark Hot Pink', 217, 1, 102),
    ('Deep Sea Blue', 1, 84, 130),
    ('Carmine', 157, 2, 22),
    ('Dark Yellow Green', 114, 143, 2),
    ('Pale Peach', 255, 229, 173),
    ('Plum Purple', 78, 5, 80),
    ('Golden Rod', 249, 188, 8),
    ('Neon Red', 255, 7, 58),
    ('Old Pink', 199, 121, 134),
    ('Very Pale Blue', 214, 255, 254),
    ('Blood Orange', 254, 75, 3),
    ('Grapefruit', 253, 89, 86),
    ('Sand Yellow', 252, 225, 102),
    ('Clay Brown', 178, 113, 61),
    ('Dark Blue Grey', 31, 59, 77),
    ('Flat Green', 105, 157, 76),
    ('Light Green Blue', 86, 252, 162),
    ('Warm Pink', 251, 85, 129),
    ('Dodger Blue', 62, 130, 252),
    ('Gross Green', 160, 191, 22),
    ('Ice', 214, 255, 250),
    ('Metallic Blue', 79, 115, 142),
    ('Pale Salmon', 255, 177, 154),
    ('Sap Green', 92, 139, 21),
    ('Algae', 84, 172, 104),
    ('Bluey Grey', 137, 160, 176),
    ('Greeny Grey', 126, 160, 122),
    ('Highlighter Green', 27, 252, 6),
    ('Light Light Blue', 202, 255, 251),
    ('Light Mint', 182, 255, 187),
    ('Raw Umber', 167, 94, 9),
    ('Vivid Blue', 21, 46, 255),
    ('Deep Lavender', 141, 94, 183),
    ('Dull Teal', 95, 158, 143),
    ('Light Greenish Blue', 99, 247, 180),
    ('Mud Green', 96, 102, 2),
    ('Pinky', 252, 134, 170),
    ('Red Wine', 140, 0, 52),
    ('Shit Green', 117, 128, 0),
    ('Tan Brown', 171, 126, 76),
    ('Darkblue', 3, 7, 100),
    ('Rosa', 254, 134, 164),
    ('Lipstick', 213, 23, 78),
    ('Pale Mauve', 254, 208, 252),
    ('Claret', 104, 0, 24),
    ('Dandelion', 254, 223, 8),
    ('Orangered', 254, 66, 15),
    ('Poop Green', 111, 124, 0),
    ('Ruby', 202, 1, 71),
    ('Dark', 27, 36, 49),
    ('Greenish Turquoise', 0, 251, 176),
    ('Pastel Red', 219, 88, 86),
    ('Piss Yellow', 221, 214, 24),
    ('Bright Cyan', 65, 253, 254),
    ('Dark Coral', 207, 82, 78),
    ('Algae Green', 33, 195, 111),
    ('Darkish Red', 169, 3, 8),
    ('Reddy Brown', 110, 16, 5),
    ('Blush Pink', 254, 130, 140),
    ('Camouflage Green', 75, 97, 19),
    ('Lawn Green', 77, 164, 9),
    ('Putty', 190, 174, 138),
    ('Vibrant Blue', 3, 57, 248),
    ('Dark Sand', 168, 143, 89),
    ('Purple/Blue', 93, 33, 208),
    ('Saffron', 254, 178, 9),
    ('Twilight', 78, 81, 139),
    ('Warm Brown', 150, 78, 2),
    ('Bluegrey', 133, 163, 178),
    ('Bubble Gum Pink', 255, 105, 175),
    ('Duck Egg Blue', 195, 251, 244),
    ('Greenish Cyan', 42, 254, 183),
    ('Petrol', 0, 95, 106),
    ('Royal', 12, 23, 147),
    ('Butter', 255, 255, 129),
    ('Dusty Orange', 240, 131, 58),
    ('Off Yellow', 241, 243, 63),
    ('Pale Olive Green', 177, 210, 123),
    ('Orangish', 252, 130, 74),
    ('Leaf', 113, 170, 52),
    ('Light Blue Grey', 183, 201, 226),
    ('Dried Blood', 75, 1, 1),
    ('Lightish Purple', 165, 82, 230),
    ('Rusty Red', 175, 47, 13),
    ('Lavender Blue', 139, 136, 248),
    ('Light Grass Green', 154, 247, 100),
    ('Light Mint Green', 166, 251, 178),
    ('Sunflower', 255, 197, 18),
    ('Velvet', 117, 8, 81),
    ('Brick Orange', 193, 74, 9),
    ('Lightish Red', 254, 47, 74),
    ('Pure Blue', 2, 3, 226),
    ('Twilight Blue', 10, 67, 122),
    ('Violet Red', 165, 0, 85),
    ('Yellowy Brown', 174, 139, 12),
    ('Carnation', 253, 121, 143),
    ('Muddy Yellow', 191, 172, 5),
    ('Dark Seafoam Green', 62, 175, 118),
    ('Deep Rose', 199, 71, 103),
    ('Dusty Red', 185, 72, 78),
    ('Grey/Blue', 100, 125, 142),
    ('Lemon Lime', 191, 254, 40),
    ('Purple/Pink', 215, 37, 222),
    ('Brown Yellow', 178, 151, 5),
    ('Purple Brown', 103, 58, 63),
    ('Wisteria', 168, 125, 194),
    ('Banana Yellow', 250, 254, 75),
    ('Lipstick Red', 192, 2, 47),
    ('Water Blue', 14, 135, 204),
    ('Brown Grey', 141, 132, 104),
    ('Vibrant Purple', 173, 3, 222),
    ('Baby Green', 140, 255, 158),
    ('Barf Green', 148, 172, 2),
    ('Eggshell Blue', 196, 255, 247),
    ('Sandy Yellow', 253, 238, 115),
    ('Cool Green', 51, 184, 100),
    ('Pale', 255, 249, 208),
    ('Blue/Grey', 117, 141, 163),
    ('Hot Magenta', 245, 4, 201),
    ('Greyblue', 119, 161, 181),
    ('Purpley', 135, 86, 228),
    ('Baby Shit Green', 136, 151, 23),
    ('Brownish Pink', 194, 126, 121),
    ('Dark Aquamarine', 1, 115, 113),
    ('Diarrhea', 159, 131, 3),
    ('Light Mustard', 247, 213, 96),
    ('Pale Sky Blue', 189, 246, 254),
    ('Turtle Green', 117, 184, 79),
    ('Bright Olive', 156, 187, 4),
    ('Dark Grey Blue', 41, 70, 91),
    ('Greeny Brown', 105, 96, 6),
    ('Lemon Green', 173, 248, 2),
    ('Light Periwinkle', 193, 198, 252),
    ('Seaweed Green', 53, 173, 107),
    ('Sunshine Yellow', 255, 253, 55),
    ('Ugly Purple', 164, 66, 160),
    ('Medium Pink', 243, 97, 150),
    ('Puke Brown', 148, 119, 6),
    ('Very Light Pink', 255, 244, 242),
    ('Viridian', 30, 145, 103),
    ('Bile', 181, 195, 6),
    ('Faded Yellow', 254, 255, 127),
    ('Very Pale Green', 207, 253, 188),
    ('Vibrant Green', 10, 221, 8),
    ('Bright Lime', 135, 253, 5),
    ('Spearmint', 30, 248, 118),
    ('Light Aquamarine', 123, 253, 199),
    ('Light Sage', 188, 236, 172),
    ('Yellowgreen', 187, 249, 15),
    ('Baby Poo', 171, 144, 4),
    ('Dark Seafoam', 31, 181, 122),
    ('Deep Teal', 0, 85, 90),
    ('Heather', 164, 132, 172),
    ('Rust Orange', 196, 85, 8),
    ('Dirty Blue', 63, 130, 157),
    ('Fern Green', 84, 141, 68),
    ('Bright Lilac', 201, 94, 251),
    ('Weird Green', 58, 229, 127),
    ('Peacock Blue', 1, 103, 149),
    ('Avocado Green', 135, 169, 34),
    ('Faded Orange', 240, 148, 77),
    ('Grape Purple', 93, 20, 81),
    ('Hot Green', 37, 255, 41),
    ('Lime Yellow', 208, 254, 29),
    ('Mango', 255, 166, 43),
    ('Shamrock', 1, 180, 76),
    ('Bubblegum', 255, 108, 181),
    ('Purplish Brown', 107, 66, 71),
    ('Vomit Yellow', 199, 193, 12),
    ('Pale Cyan', 183, 255, 250),
    ('Key Lime', 174, 255, 110),
    ('Tomato Red', 236, 45, 1),
    ('Lightgreen', 118, 255, 123),
    ('Merlot', 115, 0, 57),
    ('Night Blue', 4, 3, 72),
    ('Purpleish Pink', 223, 78, 200),
    ('Apple', 110, 203, 60),
    ('Baby Poop Green', 143, 152, 5),
    ('Green Apple', 94, 220, 31),
    ('Heliotrope', 217, 79, 245),
    ('Yellow/Green', 200, 253, 61),
    ('Almost Black', 7, 13, 13),
    ('Cool Blue', 73, 132, 184),
    ('Leafy Green', 81, 183, 59),
    ('Mustard Brown', 172, 126, 4),
    ('Dusk', 78, 84, 129),
    ('Dull Brown', 135, 110, 75),
    ('Frog Green', 88, 188, 8),
    ('Vivid Green', 47, 239, 16),
    ('Bright Light Green', 45, 254, 84),
    ('Fluro Green', 10, 255, 2),
    ('Kiwi', 156, 239, 67),
    ('Seaweed', 24, 209, 123),
    ('Navy Green', 53, 83, 10),
    ('Ultramarine Blue', 24, 5, 219),
    ('Iris', 98, 88, 196),
    ('Pastel Orange', 255, 150, 79),
    ('Yellowish Orange', 255, 171, 15),
    ('Perrywinkle', 143, 140, 231),
    ('Tealish', 36, 188, 168),
    ('Dark Plum', 63, 1, 44),
    ('Pear', 203, 248, 95),
    ('Pinkish Orange', 255, 114, 76),
    ('Midnight Purple', 40, 1, 55),
    ('Light Urple', 179, 111, 246),
    ('Dark Mint', 72, 192, 114),
    ('Greenish Tan', 188, 203, 122),
    ('Light Burgundy', 168, 65, 91),
    ('Turquoise Blue', 6, 177, 196),
    ('Ugly Pink', 205, 117, 132),
    ('Sandy', 241, 218, 122),
    ('Electric Pink', 255, 4, 144),
    ('Muted Purple', 128, 91, 135),
    ('Mid Green', 80, 167, 71),
    ('Greyish', 168, 164, 149),
    ('Neon Yellow', 207, 255, 4),
    ('Banana', 255, 255, 126),
    ('Carnation Pink', 255, 127, 167),
    ('Tomato', 239, 64, 38),
    ('Sea', 60, 153, 146),
    ('Muddy Brown', 136, 104, 6),
    ('Turquoise Green', 4, 244, 137),
    ('Buff', 254, 246, 158),
    ('Fawn', 207, 175, 123),
    ('Muted Blue', 59, 113, 159),
    ('Pale Rose', 253, 193, 197),
    ('Dark Mint Green', 32, 192, 115),
    ('Amethyst', 155, 95, 192),
    ('Blue/Green', 15, 155, 142),
    ('Chestnut', 116, 40, 2),
    ('Sick Green', 157, 185, 44),
    ('Pea', 164, 191, 32),
    ('Rusty Orange', 205, 89, 9),
    ('Stone', 173, 165, 135),
    ('Rose Red', 190, 1, 60),
    ('Pale Aqua', 184, 255, 235),
    ('Deep Orange', 220, 77, 1),
    ('Earth', 162, 101, 62),
    ('Mossy Green', 99, 139, 39),
    ('Grassy Green', 65, 156, 3),
    ('Pale Lime Green', 177, 255, 101),
    ('Light Grey Blue', 157, 188, 212),
    ('Pale Grey', 253, 253, 254),
    ('Asparagus', 119, 171, 86),
    ('Blueberry', 70, 65, 150),
    ('Purple Red', 153, 1, 71),
    ('Pale Lime', 190, 253, 115),
    ('Greenish Teal', 50, 191, 132),
    ('Caramel', 175, 111, 9),
    ('Deep Magenta', 160, 2, 92),
    ('Light Peach', 255, 216, 177),
    ('Milk Chocolate', 127, 78, 30),
    ('Ocher', 191, 155, 12),
    ('Off Green', 107, 163, 83),
    ('Purply Pink', 240, 117, 230),
    ('Lightblue', 123, 200, 246),
    ('Dusky Blue', 71, 95, 148),
    ('Golden', 245, 191, 3),
    ('Light Beige', 255, 254, 182),
    ('Butter Yellow', 255, 253, 116),
    ('Dusky Purple', 137, 91, 123),
    ('French Blue', 67, 107, 173),
    ('Ugly Yellow', 208, 193, 1),
    ('Greeny Yellow', 198, 248, 8),
    ('Orangish Red', 244, 54, 5),
    ('Shamrock Green', 2, 193, 77),
    ('Orangish Brown', 178, 95, 3),
    ('Tree Green', 42, 126, 25),
    ('Deep Violet', 73, 6, 72),
    ('Gunmetal', 83, 98, 103),
    ('Blue/Purple', 90, 6, 239),
    ('Cherry', 207, 2, 52),
    ('Sandy Brown', 196, 166, 97),
    ('Warm Grey', 151, 138, 132),
    ('Dark Indigo', 31, 9, 84),
    ('Midnight', 3, 1, 45),
    ('Bluey Green', 43, 177, 121),
    ('Grey Pink', 195, 144, 155),
    ('Soft Purple', 166, 111, 181),
    ('Blood', 119, 0, 1),
    ('Brown Red', 146, 43, 5),
    ('Medium Grey', 125, 127, 124),
    ('Berry', 153, 15, 75),
    ('Poo', 143, 115, 3),
    ('Purpley Pink', 200, 60, 185),
    ('Light Salmon', 254, 169, 147),
    ('Snot', 172, 187, 13),
    ('Easter Purple', 192, 113, 254),
    ('Light Yellow Green', 204, 253, 127),
    ('Dark Navy Blue', 0, 2, 46),
    ('Drab', 130, 131, 68),
    ('Light Rose', 255, 197, 203),
    ('Rouge', 171, 18, 57),
    ('Purplish Red', 176, 5, 75),
    ('Slime Green', 153, 204, 4),
    ('Baby Poop', 147, 124, 0),
    ('Irish Green', 1, 149, 41),
    ('Pink/Purple', 239, 29, 231),
    ('Dark Navy', 0, 4, 53),
    ('Greeny Blue', 66, 179, 149),
    ('Light Plum', 157, 87, 131),
    ('Pinkish Grey', 200, 172, 169),
    ('Dirty Orange', 200, 118, 6),
    ('Rust Red', 170, 39, 4),
    ('Pale Lilac', 228, 203, 255),
    ('Orangey Red', 250, 66, 36),
    ('Primary Blue', 8, 4, 249),
    ('Kermit Green', 92, 178, 0),
    ('Brownish Purple', 118, 66, 78),
    ('Murky Green', 108, 122, 14),
    ('Wheat', 251, 221, 126),
    ('Very Dark Purple', 42, 1, 52),
    ('Bottle Green', 4, 74, 5),
    ('Watermelon', 253, 70, 89),
    ('Deep Sky Blue', 13, 117, 248),
    ('Fire Engine Red', 254, 0, 2),
    ('Yellow Ochre', 203, 157, 6),
    ('Pumpkin Orange', 251, 125, 7),
    ('Pale Olive', 185, 204, 129),
    ('Light Lilac', 237, 200, 255),
    ('Lightish Green', 97, 225, 96),
    ('Carolina Blue', 138, 184, 254),
    ('Mulberry', 146, 10, 78),
    ('Shocking Pink', 254, 2, 162),
    ('Auburn', 154, 48, 1),
    ('Bright Lime Green', 101, 254, 8),
    ('Celadon', 190, 253, 183),
    ('Pinkish Brown', 177, 114, 97),
    ('Poo Brown', 136, 95, 1),
    ('Bright Sky Blue', 2, 204, 254),
    ('Celery', 193, 253, 149),
    ('Dirt Brown', 131, 101, 57),
    ('Strawberry', 251, 41, 67),
    ('Dark Lime', 132, 183, 1),
    ('Copper', 182, 99, 37),
    ('Medium Brown', 127, 81, 18),
    ('Muted Green', 95, 160, 82),
    ("Robin'S Egg", 109, 237, 253),
    ('Bright Aqua', 11, 249, 234),
    ('Bright Lavender', 199, 96, 255),
    ('Ivory', 255, 255, 203),
    ('Very Light Purple', 246, 206, 252),
    ('Light Navy', 21, 80, 132),
    ('Pink Red', 245, 5, 79),
    ('Olive Brown', 100, 84, 3),
    ('Poop Brown', 122, 89, 1),
    ('Mustard Green', 168, 181, 4),
    ('Ocean Green', 61, 153, 115),
    ('Very Dark Blue', 0, 1, 51),
    ('Dusty Green', 118, 169, 115),
    ('Light Navy Blue', 46, 90, 136),
    ('Minty Green', 11, 247, 125),
    ('Adobe', 189, 108, 72),
    ('Barney', 172, 29, 184),
    ('Jade Green', 43, 175, 106),
    ('Bright Light Blue', 38, 247, 253),
    ('Light Lime', 174, 253, 108),
    ('Dark Khaki', 155, 143, 85),
    ('Orange Yellow', 255, 173, 1),
    ('Ocre', 198, 156, 4),
    ('Maize', 244, 208, 84),
    ('Faded Pink', 222, 157, 172),
    ('British Racing Green', 5, 72, 13),
    ('Sandstone', 201, 174, 116),
    ('Mud Brown', 96, 70, 15),
    ('Light Sea Green', 152, 246, 176),
    ('Robin Egg Blue', 138, 241, 254),
    ('Aqua Marine', 46, 232, 187),
    ('Dark Sea Green', 17, 135, 93),
    ('Soft Pink', 253, 176, 192),
    ('Orangey Brown', 177, 96, 2),
    ('Cherry Red', 247, 2, 42),
    ('Burnt Yellow', 213, 171, 9),
    ('Brownish Grey', 134, 119, 95),
    ('Camel', 198, 159, 89),
    ('Purplish Grey', 122, 104, 127),
    ('Marine', 4, 46, 96),
    ('Greyish Pink', 200, 141, 148),
    ('Pale Turquoise', 165, 251, 213),
    ('Pastel Yellow', 255, 254, 113),
    ('Bluey Purple', 98, 65, 199),
    ('Canary Yellow', 255, 254, 64),
    ('Faded Red', 211, 73, 78),
    ('Sepia', 152, 94, 43),
    ('Coffee', 166, 129, 76),
    ('Bright Magenta', 255, 8, 232),
    ('Mocha', 157, 118, 81),
    ('Ecru', 254, 255, 202),
    ('Purpleish', 152, 86, 141),
    ('Cranberry', 158, 0, 58),
    ('Darkish Green', 40, 124, 55),
    ('Brown Orange', 185, 105, 2),
    ('Dusky Rose', 186, 104, 115),
    ('Melon', 255, 120, 85),
    ('Sickly Green', 148, 178, 28),
    ('Silver', 197, 201, 199),
    ('Purply Blue', 102, 26, 238),
    ('Purpleish Blue', 97, 64, 239),
    ('Hospital Green', 155, 229, 170),
    ('Shit Brown', 123, 88, 4),
    ('Mid Blue', 39, 106, 179),
    ('Amber', 254, 179, 8),
    ('Easter Green', 140, 253, 126),
    ('Soft Blue', 100, 136, 234),
    ('Cerulean Blue', 5, 110, 238),
    ('Golden Brown', 178, 122, 1),
    ('Bright Turquoise', 15, 254, 249),
    ('Red Pink', 250, 42, 85),
    ('Red Purple', 130, 7, 71),
    ('Greyish Brown', 122, 106, 79),
    ('Vermillion', 244, 50, 12),
    ('Russet', 161, 57, 5),
    ('Steel Grey', 111, 130, 138),
    ('Lighter Purple', 165, 90, 244),
    ('Bright Violet', 173, 10, 253),
    ('Prussian Blue', 0, 69, 119),
    ('Slate Green', 101, 141, 109),
    ('Dirty Pink', 202, 123, 128),
    ('Dark Blue Green', 0, 82, 73),
    ('Pine', 43, 93, 52),
    ('Yellowy Green', 191, 241, 40),
    ('Dark Gold', 181, 148, 16),
    ('Bluish', 41, 118, 187),
    ('Darkish Blue', 1, 65, 130),
    ('Dull Red', 187, 63, 63),
    ('Pinky Red', 252, 38, 71),
    ('Bronze', 168, 121, 0),
    ('Pale Teal', 130, 203, 178),
    ('Military Green', 102, 124, 62),
    ('Barbie Pink', 254, 70, 165),
    ('Bubblegum Pink', 254, 131, 204),
    ('Pea Soup Green', 148, 166, 23),
    ('Dark Mustard', 168, 137, 5),
    ('Shit', 127, 95, 0),
    ('Medium Purple', 158, 67, 162),
    ('Very Dark Green', 6, 46, 3),
    ('Dirt', 138, 110, 69),
    ('Dusky Pink', 204, 122, 139),
    ('Red Violet', 158, 1, 104),
    ('Lemon Yellow', 253, 255, 56),
    ('Pistachio', 192, 250, 139),
    ('Dull Yellow', 238, 220, 91),
    ('Dark Lime Green', 126, 189, 1),
    ('Denim Blue', 59, 91, 146),
    ('Teal Blue', 1, 136, 159),
    ('Lightish Blue', 61, 122, 253),
    ('Purpley Blue', 95, 52, 231),
    ('Light Indigo', 109, 90, 207),
    ('Swamp Green', 116, 133, 0),
    ('Brown Green', 112, 108, 17),
    ('Dark Maroon', 60, 0, 8),
    ('Hot Purple', 203, 0, 245),
    ('Dark Forest Green', 0, 45, 4),
    ('Faded Blue', 101, 140, 187),
    ('Drab Green', 116, 149, 81),
    ('Light Lime Green', 185, 255, 102),
    ('Snot Green', 157, 193, 0),
    ('Yellowish', 250, 238, 102),
    ('Light Blue Green', 126, 251, 179),
    ('Bordeaux', 123, 0, 44),
    ('Light Mauve', 194, 146, 161),
    ('Ocean', 1, 123, 146),
    ('Marigold', 252, 192, 6),
    ('Muddy Green', 101, 116, 50),
    ('Dull Orange', 216, 134, 59),
    ('Steel', 115, 133, 149),
    ('Electric Purple', 170, 35, 255),
    ('Fluorescent Green', 8, 255, 8),
    ('Yellowish Brown', 155, 122, 1),
    ('Blush', 242, 158, 142),
    ('Soft Green', 111, 194, 118),
    ('Bright Orange', 255, 91, 0),
    ('Lemon', 253, 255, 82),
    ('Purple Grey', 134, 111, 133),
    ('Acid Green', 143, 254, 9),
    ('Pale Lavender', 238, 207, 254),
    ('Violet Blue', 81, 10, 201),
    ('Light Forest Green', 79, 145, 83),
    ('Burnt Red', 159, 35, 5),
    ('Khaki Green', 114, 134, 57),
    ('Cerise', 222, 12, 98),
    ('Faded Purple', 145, 110, 153),
    ('Apricot', 255, 177, 109),
    ('Dark Olive Green', 60, 77, 3),
    ('Grey Brown', 127, 112, 83),
    ('Green Grey', 119, 146, 111),
    ('True Blue', 1, 15, 204),
    ('Pale Violet', 206, 174, 250),
    ('Periwinkle Blue', 143, 153, 251),
    ('Light Sky Blue', 198, 252, 255),
    ('Blurple', 85, 57, 204),
    ('Green Brown', 84, 78, 3),
    ('Bluegreen', 1, 122, 121),
    ('Bright Teal', 1, 249, 198),
    ('Brownish Yellow', 201, 176, 3),
    ('Pea Soup', 146, 153, 1),
    ('Forest', 11, 85, 9),
    ('Barney Purple', 160, 4, 152),
    ('Ultramarine', 32, 0, 177),
    ('Purplish', 148, 86, 140),
    ('Puke Yellow', 194, 190, 14),
    ('Bluish Grey', 116, 139, 151),
    ('Dark Periwinkle', 102, 95, 209),
    ('Dark Lilac', 156, 109, 165),
    ('Reddish', 196, 66, 64),
    ('Light Maroon', 162, 72, 87),
    ('Dusty Purple', 130, 95, 135),
    ('Terra Cotta', 201, 100, 59),
    ('Avocado', 144, 177, 52),
    ('Marine Blue', 1, 56, 106),
    ('Teal Green', 37, 163, 111),
    ('Slate Grey', 89, 101, 109),
    ('Lighter Green', 117, 253, 99),
    ('Electric Green', 33, 252, 13),
    ('Dusty Blue', 90, 134, 173),
    ('Golden Yellow', 254, 198, 21),
    ('Bright Yellow', 255, 253, 1),
    ('Light Lavender', 223, 197, 254),
    ('Umber', 178, 100, 0),
    ('Poop', 127, 94, 0),
    ('Dark Peach', 222, 126, 93),
    ('Jungle Green', 4, 130, 67),
    ('Eggshell', 255, 255, 212),
    ('Denim', 59, 99, 140),
    ('Yellow Brown', 183, 148, 0),
    ('Dull Purple', 132, 89, 126),
    ('Chocolate Brown', 65, 25, 0),
    ('Wine Red', 123, 3, 35),
    ('Neon Blue', 4, 217, 255),
    ('Dirty Green', 102, 126, 44),
    ('Light Tan', 251, 238, 172),
    ('Ice Blue', 215, 255, 254),
    ('Cadet Blue', 78, 116, 150),
    ('Dark Mauve', 135, 76, 98),
    ('Very Light Blue', 213, 255, 255),
    ('Grey Purple', 130, 109, 140),
    ('Pastel Pink', 255, 186, 205),
    ('Very Light Green', 209, 255, 189),
    ('Dark Sky Blue', 68, 142, 228),
    ('Evergreen', 5, 71, 42),
    ('Dull Pink', 213, 134, 157),
    ('Aubergine', 61, 7, 52),
    ('Mahogany', 74, 1, 0),
    ('Reddish Orange', 248, 72, 28),
    ('Deep Green', 2, 89, 15),
    ('Vomit Green', 137, 162, 3),
    ('Purple Pink', 224, 63, 216),
    ('Dusty Pink', 213, 138, 148),
    ('Faded Green', 123, 178, 116),
    ('Camo Green', 82, 101, 37),
    ('Pinky Purple', 201, 76, 190),
    ('Pink Purple', 219, 75, 218),
    ('Brownish Red', 158, 54, 35),
    ('Dark Rose', 181, 72, 93),
    ('Mud', 115, 92, 18),
    ('Brownish', 156, 109, 87),
    ('Emerald Green', 2, 143, 30),
    ('Pale Brown', 177, 145, 110),
    ('Dull Blue', 73, 117, 156),
    ('Burnt Umber', 160, 69, 14),
    ('Medium Green', 57, 173, 72),
    ('Clay', 182, 106, 80),
    ('Light Aqua', 140, 255, 219),
    ('Light Olive Green', 164, 190, 92),
    ('Brownish Orange', 203, 119, 35),
    ('Dark Aqua', 5, 105, 107),
    ('Purplish Pink', 206, 93, 174),
    ('Dark Salmon', 200, 90, 83),
    ('Greenish Grey', 150, 174, 141),
    ('Jade', 31, 167, 116),
    ('Ugly Green', 122, 151, 3),
    ('Dark Beige', 172, 147, 98),
    ('Emerald', 1, 160, 73),
    ('Pale Red', 217, 84, 77),
    ('Light Magenta', 250, 95, 247),
    ('Sky', 130, 202, 252),
    ('Light Cyan', 172, 255, 252),
    ('Yellow Orange', 252, 176, 1),
    ('Reddish Purple', 145, 9, 81),
    ('Reddish Pink', 254, 44, 84),
    ('Orchid', 200, 117, 196),
    ('Dirty Yellow', 205, 197, 10),
    ('Orange Red', 253, 65, 30),
    ('Deep Red', 154, 2, 0),
    ('Orange Brown', 190, 100, 0),
    ('Cobalt Blue', 3, 10, 167),
    ('Neon Pink', 254, 1, 154),
    ('Rose Pink', 247, 135, 154),
    ('Greyish Purple', 136, 113, 145),
    ('Raspberry', 176, 1, 73),
    ('Aqua Green', 18, 225, 147),
    ('Salmon Pink', 254, 123, 124),
    ('Tangerine', 255, 148, 8),
    ('Brownish Green', 106, 110, 9),
    ('Red Brown', 139, 46, 22),
    ('Greenish Brown', 105, 97, 18),
    ('Pumpkin', 225, 119, 1),
    ('Pine Green', 10, 72, 30),
    ('Charcoal', 52, 56, 55),
    ('Baby Pink', 255, 183, 206),
    ('Cornflower', 106, 121, 247),
    ('Blue Violet', 93, 6, 233),
    ('Chocolate', 61, 28, 2),
    ('Greyish Green', 130, 166, 125),
    ('Scarlet', 190, 1, 25),
    ('Green Yellow', 201, 255, 39),
    ('Dark Olive', 55, 62, 2),
    ('Sienna', 169, 86, 30),
    ('Pastel Purple', 202, 160, 255),
    ('Terracotta', 202, 102, 65),
    ('Aqua Blue', 2, 216, 233),
    ('Sage Green', 136, 179, 120),
    ('Blood Red', 152, 0, 2),
    ('Deep Pink', 203, 1, 98),
    ('Grass', 92, 172, 45),
    ('Moss', 118, 153, 88),
    ('Pastel Blue', 162, 191, 254),
    ('Bluish Green', 16, 166, 116),
    ('Green Blue', 6, 180, 139),
    ('Dark Tan', 175, 136, 74),
    ('Greenish Blue', 11, 139, 135),
    ('Pale Orange', 255, 167, 86),
    ('Vomit', 162, 164, 21),
    ('Forrest Green', 21, 68, 6),
    ('Dark Lavender', 133, 103, 152),
    ('Dark Violet', 52, 1, 63),
    ('Purple Blue', 99, 45, 233),
    ('Dark Cyan', 10, 136, 138),
    ('Olive Drab', 111, 118, 50),
    ('Pinkish', 212, 106, 126),
    ('Cobalt', 30, 72, 143),
    ('Neon Purple', 188, 19, 254),
    ('Light Turquoise', 126, 244, 204),
    ('Apple Green', 118, 205, 38),
    ('Dull Green', 116, 166, 98),
    ('Wine', 128, 1, 63),
    ('Powder Blue', 177, 209, 252),
    ('Off White', 255, 255, 228),
    ('Electric Blue', 6, 82, 255),
    ('Dark Turquoise', 4, 92, 90),
    ('Blue Purple', 87, 41, 206),
    ('Azure', 6, 154, 243),
    ('Bright Red', 255, 0, 13),
    ('Pinkish Red', 241, 12, 69),
    ('Cornflower Blue', 81, 112, 215),
    ('Light Olive', 172, 191, 105),
    ('Grape', 108, 52, 97),
    ('Greyish Blue', 94, 129, 157),
    ('Purplish Blue', 96, 30, 249),
    ('Yellowish Green', 176, 221, 22),
    ('Greenish Yellow', 205, 253, 2),
    ('Medium Blue', 44, 111, 187),
    ('Dusty Rose', 192, 115, 122),
    ('Light Violet', 214, 180, 252),
    ('Midnight Blue', 2, 0, 53),
    ('Bluish Purple', 112, 59, 231),
    ('Red Orange', 253, 60, 6),
    ('Dark Magenta', 150, 0, 86),
    ('Greenish', 64, 163, 104),
    ('Ocean Blue', 3, 113, 156),
    ('Coral', 252, 90, 80),
    ('Cream', 255, 255, 194),
    ('Reddish Brown', 127, 43, 10),
    ('Burnt Sienna', 176, 78, 15),
    ('Brick', 160, 54, 35),
    ('Sage', 135, 174, 115),
    ('Grey Green', 120, 155, 115),
    ('White', 255, 255, 255),
    ("Robin'S Egg Blue", 152, 239, 249),
    ('Moss Green', 101, 139, 56),
    ('Steel Blue', 90, 125, 154),
    ('Eggplant', 56, 8, 53),
    ('Light Yellow', 255, 254, 122),
    ('Leaf Green', 92, 169, 4),
    ('Light Grey', 216, 220, 214),
    ('Puke', 165, 165, 2),
    ('Pinkish Purple', 214, 72, 215),
    ('Sea Blue', 4, 116, 149),
    ('Pale Purple', 183, 144, 212),
    ('Slate Blue', 91, 124, 153),
    ('Blue Grey', 96, 124, 142),
    ('Hunter Green', 11, 64, 8),
    ('Fuchsia', 237, 13, 217),
    ('Crimson', 140, 0, 15),
    ('Pale Yellow', 255, 255, 132),
    ('Ochre', 191, 144, 5),
    ('Mustard Yellow', 210, 189, 10),
    ('Light Red', 255, 71, 76),
    ('Cerulean', 4, 133, 209),
    ('Pale Pink', 255, 207, 220),
    ('Deep Blue', 4, 2, 115),
    ('Rust', 168, 60, 9),
    ('Light Teal', 144, 228, 193),
    ('Slate', 81, 101, 114),
    ('Goldenrod', 250, 194, 5),
    ('Dark Yellow', 213, 182, 10),
    ('Dark Grey', 54, 55, 55),
    ('Army Green', 75, 93, 22),
    ('Grey Blue', 107, 139, 164),
    ('Seafoam', 128, 249, 173),
    ('Puce', 165, 126, 82),
    ('Spring Green', 169, 249, 113),
    ('Dark Orange', 198, 81, 2),
    ('Sand', 226, 202, 118),
    ('Pastel Green', 176, 255, 157),
    ('Mint', 159, 254, 176),
    ('Light Orange', 253, 170, 72),
    ('Bright Pink', 254, 1, 177),
    ('Chartreuse', 193, 248, 10),
    ('Deep Purple', 54, 1, 63),
    ('Dark Brown', 52, 28, 2),
    ('Taupe', 185, 162, 129),
    ('Pea Green', 142, 171, 18),
    ('Puke Green', 154, 174, 7),
    ('Kelly Green', 2, 171, 46),
    ('Seafoam Green', 122, 249, 171),
    ('Blue Green', 19, 126, 109),
    ('Khaki', 170, 166, 98),
    ('Burgundy', 97, 0, 35),
    ('Dark Teal', 1, 77, 78),
    ('Brick Red', 143, 20, 2),
    ('Royal Purple', 75, 0, 110),
    ('Plum', 88, 15, 65),
    ('Mint Green', 143, 255, 159),
    ('Gold', 219, 180, 12),
    ('Baby Blue', 162, 207, 254),
    ('Yellow Green', 192, 251, 45),
    ('Bright Purple', 190, 3, 253),
    ('Dark Red', 132, 0, 0),
    ('Pale Blue', 208, 254, 254),
    ('Grass Green', 63, 155, 11),
    ('Navy', 1, 21, 62),
    ('Aquamarine', 4, 216, 178),
    ('Burnt Orange', 192, 78, 1),
    ('Neon Green', 12, 255, 12),
    ('Bright Blue', 1, 101, 252),
    ('Rose', 207, 98, 117),
    ('Light Pink', 255, 209, 223),
    ('Mustard', 206, 179, 1),
    ('Indigo', 56, 2, 130),
    ('Lime', 170, 255, 50),
    ('Sea Green', 83, 252, 161),
    ('Periwinkle', 142, 130, 254),
    ('Dark Pink', 203, 65, 107),
    ('Olive Green', 103, 122, 4),
    ('Peach', 255, 176, 124),
    ('Pale Green', 199, 253, 181),
    ('Light Brown', 173, 129, 80),
    ('Hot Pink', 255, 2, 141),
    ('Black', 0, 0, 0),
    ('Lilac', 206, 162, 253),
    ('Navy Blue', 0, 17, 70),
    ('Royal Blue', 5, 4, 170),
    ('Beige', 230, 218, 166),
    ('Salmon', 255, 121, 108),
    ('Olive', 110, 117, 14),
    ('Maroon', 101, 0, 33),
    ('Bright Green', 1, 255, 7),
    ('Dark Purple', 53, 6, 62),
    ('Mauve', 174, 113, 129),
    ('Forest Green', 6, 71, 12),
    ('Aqua', 19, 234, 201),
    ('Cyan', 0, 255, 255),
    ('Tan', 209, 178, 111),
    ('Dark Blue', 0, 3, 91),
    ('Lavender', 199, 159, 239),
    ('Turquoise', 6, 194, 172),
    ('Dark Green', 3, 53, 0),
    ('Violet', 154, 14, 234),
    ('Light Purple', 191, 119, 246),
    ('Lime Green', 137, 254, 5),
    ('Grey', 146, 149, 145),
    ('Sky Blue', 117, 187, 253),
    ('Yellow', 255, 255, 20),
    ('Magenta', 194, 0, 120),
    ('Light Green', 150, 249, 123),
    ('Orange', 249, 115, 6),
    ('Teal', 2, 147, 134),
    ('Light Blue', 149, 208, 252),
    ('Red', 229, 0, 0),
    ('Brown', 101, 55, 0),
    ('Pink', 255, 129, 192),
    ('Blue', 3, 67, 223),
    ('Green', 21, 176, 26),
    ('Purple', 126, 30, 156),
)

@functools.lru_cache(maxsize=1)
def _get_xkcd_lab_colors() -> tuple[tuple[str, ...], np.ndarray]:
    """Get XKCD colors converted to LAB space. Cached after first call."""
    names = tuple(name for name, *_ in XKCD_COLORS)
    rgb_array = np.array([[r, g, b] for _, r, g, b in XKCD_COLORS])
    labs = rgb_to_lab(rgb_array)  # Single batched call instead of 949 individual calls
    return names, labs


# =============================================================================
# Color Conversion
# =============================================================================

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array (0-255) to LAB color space."""
    rgb_norm = rgb.astype(np.float64) / 255.0

    # Apply gamma correction
    mask = rgb_norm > 0.04045
    rgb_linear = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

    # RGB to XYZ matrix
    r, g, b = rgb_linear[:, 0], rgb_linear[:, 1], rgb_linear[:, 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # XYZ to LAB (D65 reference white)
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x, y, z = x / xn, y / yn, z / zn

    epsilon = 0.008856
    kappa = 903.3
    fx = np.where(x > epsilon, x ** (1/3), (kappa * x + 16) / 116)
    fy = np.where(y > epsilon, y ** (1/3), (kappa * y + 16) / 116)
    fz = np.where(z > epsilon, z ** (1/3), (kappa * z + 16) / 116)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)

    return np.column_stack([L, a, b_val])


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB array to RGB (0-255)."""
    if lab.ndim == 1:
        lab = lab.reshape(1, -1)

    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    epsilon = 0.008856
    kappa = 903.3

    x = np.where(fx**3 > epsilon, fx**3, (116 * fx - 16) / kappa)
    y = np.where(L > kappa * epsilon, ((L + 16) / 116) ** 3, L / kappa)
    z = np.where(fz**3 > epsilon, fz**3, (116 * fz - 16) / kappa)

    x *= 0.95047
    z *= 1.08883

    r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314
    g = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560
    b_out = x * 0.0556434 - y * 0.2040259 + z * 1.0572252

    rgb_linear = np.column_stack([r, g, b_out])
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * np.power(np.clip(rgb_linear, 0, None), 1/2.4) - 0.055, 12.92 * rgb_linear)

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def lab_to_hex(lab: np.ndarray) -> str:
    """Convert LAB to hex string."""
    rgb = lab_to_rgb(lab)
    if rgb.ndim > 1:
        rgb = rgb[0]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def lab_to_rgb_tuple(lab: np.ndarray) -> tuple[int, int, int]:
    """Convert LAB to RGB tuple."""
    rgb = lab_to_rgb(lab)
    if rgb.ndim > 1:
        rgb = rgb[0]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


# =============================================================================
# Color Utilities
# =============================================================================

def compute_chroma(lab: np.ndarray) -> float:
    """Compute chroma (saturation) from LAB coordinates."""
    return float(np.sqrt(lab[1]**2 + lab[2]**2))


def delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Compute Delta E (Euclidean distance in LAB space)."""
    return float(np.linalg.norm(lab1 - lab2))


def compute_hue(lab: np.ndarray) -> float:
    """Compute hue angle (0-360 degrees) from LAB coordinates."""
    return float(np.degrees(np.arctan2(lab[2], lab[1])) % 360)


def circular_hue_distance(hue1: float, hue2: float) -> float:
    """Compute minimum angular distance between two hues (0-180)."""
    diff = abs(hue1 - hue2)
    return min(diff, 360 - diff)


def encode_bins(bins: np.ndarray) -> np.ndarray:
    """
    Encode 3D bin indices to single int64 values for fast numpy operations.

    Args:
        bins: Array of shape (..., 3) with int32 bin indices (L, a, b)

    Returns:
        Array of shape (...) with int64 encoded values
    """
    # Shift to positive range (bins can be negative for a/b channels)
    # L bins: 0 to ~43, a/b bins: -56 to +55 (all fit in 8 bits after +128 shift)
    shifted = bins.astype(np.int64) + 128
    return shifted[..., 0] * 65536 + shifted[..., 1] * 256 + shifted[..., 2]


def encode_bin_key(bin_key: BinKey) -> int:
    """Encode a single bin key tuple to int64 (avoids array creation overhead)."""
    L, a, b = bin_key
    return (L + 128) * 65536 + (a + 128) * 256 + (b + 128)


def decode_bins(encoded: np.ndarray) -> np.ndarray:
    """
    Decode int64 encoded bins back to 3D bin indices.

    Args:
        encoded: Array of int64 encoded bin values

    Returns:
        Array of shape (..., 3) with int32 bin indices (L, a, b)
    """
    encoded = encoded.astype(np.int64)
    L = (encoded // 65536) - 128
    a = ((encoded % 65536) // 256) - 128
    b = (encoded % 256) - 128
    return np.stack([L, a, b], axis=-1).astype(np.int32)


# =============================================================================
# Stage 1: Data Preparation
# =============================================================================

@dataclass
class BinData:
    """Data for a single color bin."""
    lab: np.ndarray  # Representative LAB value
    count: int  # Pixel count
    positions: list[Position] = field(default_factory=list)  # (y, x) positions


@dataclass
class PreparedData:
    """Output of Stage 1: Data Preparation."""
    fine_bins: dict[BinKey, BinData]
    coarse_bins: dict[BinKey, BinData]
    fine_to_coarse: dict[BinKey, BinKey]
    coarse_to_fine: dict[BinKey, list[BinKey]]
    total_pixels: int
    image_shape: tuple[int, int]
    # Pre-built arrays for vectorized operations
    fine_binned: np.ndarray  # (h, w, 3) array of fine bin indices
    coarse_binned: np.ndarray  # (h, w, 3) array of coarse bin indices


def prepare_data(image_path: str, downscale: bool = True) -> PreparedData:
    """
    Stage 1: Load image and quantize at two scales.

    Fine scale (JND): preserves gradient steps
    Coarse scale (~5x JND): captures major color blocks

    Args:
        image_path: Path to the image file
        downscale: If True, resize images so longest edge is 256px (default: True)

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file is not a valid image or exceeds size limits
    """
    # Load image with validation
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {image_path}")
    except UnidentifiedImageError as e:
        raise ValueError(f"Not a valid image file: {e}")
    except IOError as e:
        raise ValueError(f"Could not read image: {e}")

    # Validate image dimensions (security: prevent decompression bombs)
    width, height = img.size
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions {width}x{height} exceed maximum "
            f"{MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image has {width * height:,} pixels, exceeding maximum {MAX_IMAGE_PIXELS:,}"
        )

    # Downscale if enabled and image exceeds 256px on longest edge
    if downscale and max(width, height) > 256:
        scale = 256 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    img = img.convert('RGB')
    pixels = np.array(img)
    h, w = pixels.shape[:2]

    # Convert to LAB
    lab_flat = rgb_to_lab(pixels.reshape(-1, 3))
    lab_image = lab_flat.reshape(h, w, 3)

    # Quantize at both scales
    fine_size = FINE_SCALE * JND
    coarse_size = COARSE_SCALE * JND

    fine_binned = np.round(lab_image / fine_size).astype(np.int32)
    coarse_binned = np.round(lab_image / coarse_size).astype(np.int32)

    # Vectorized bin data extraction using np.unique
    # Reshape to (n_pixels, 3) for unique operation
    fine_flat = fine_binned.reshape(-1, 3)
    coarse_flat = coarse_binned.reshape(-1, 3)

    # Get unique bins and counts for fine scale
    fine_unique, fine_inverse, fine_counts = np.unique(
        fine_flat, axis=0, return_inverse=True, return_counts=True
    )

    # Get unique bins and counts for coarse scale
    coarse_unique, coarse_inverse, coarse_counts = np.unique(
        coarse_flat, axis=0, return_inverse=True, return_counts=True
    )

    # Build fine bins dict (without positions - use arrays directly)
    fine_bins: dict[BinKey, BinData] = {}
    for i, (bin_arr, count) in enumerate(zip(fine_unique, fine_counts)):
        bin_key = tuple(bin_arr)
        fine_lab = bin_arr.astype(np.float64) * fine_size
        fine_bins[bin_key] = BinData(lab=fine_lab, count=int(count), positions=[])

    # Build coarse bins dict
    coarse_bins: dict[BinKey, BinData] = {}
    for i, (bin_arr, count) in enumerate(zip(coarse_unique, coarse_counts)):
        bin_key = tuple(bin_arr)
        coarse_lab = bin_arr.astype(np.float64) * coarse_size
        coarse_bins[bin_key] = BinData(lab=coarse_lab, count=int(count), positions=[])

    # Build scale mappings using vectorized lookup
    # For each fine bin, find corresponding coarse bin
    fine_to_coarse: dict[BinKey, BinKey] = {}
    coarse_to_fine: dict[BinKey, list[BinKey]] = {}

    for fine_bin in fine_bins:
        # Get one pixel position with this fine bin to find its coarse bin
        fine_arr = np.array(fine_bin)
        coarse_arr = np.round(fine_arr * fine_size / coarse_size).astype(np.int32)
        coarse_bin = tuple(coarse_arr)
        fine_to_coarse[fine_bin] = coarse_bin
        if coarse_bin not in coarse_to_fine:
            coarse_to_fine[coarse_bin] = []
        coarse_to_fine[coarse_bin].append(fine_bin)

    return PreparedData(
        fine_bins=fine_bins,
        coarse_bins=coarse_bins,
        fine_to_coarse=fine_to_coarse,
        coarse_to_fine=coarse_to_fine,
        total_pixels=h * w,
        image_shape=(h, w),
        fine_binned=fine_binned,
        coarse_binned=coarse_binned,
    )


# =============================================================================
# Stage 2: Feature Extraction
# =============================================================================

@dataclass
class StabilityInfo:
    """Scale stability classification for a coarse bin."""
    stability_type: str  # 'anchor', 'gradient', 'texture'
    fine_count: int  # Number of fine children
    fine_variance: float  # LAB variance of fine children
    dominant_axis: str | None = None  # 'L', 'a', 'b' if gradient


@dataclass
class ColorFamily:
    """A group of colors with similar hue."""
    hue_center: float  # 0-360
    lightness_range: tuple[float, float]
    chroma_range: tuple[float, float]
    total_coverage: float  # 0-1
    members: list[BinKey]
    is_neutral: bool = False


@dataclass
class GradientChain:
    """A detected gradient in the image."""
    stops: list[BinKey]  # Coarse bins (anchor colors only)
    fine_members: list[BinKey]  # All fine bins in the chain
    direction: str  # 'horizontal', 'vertical', 'diagonal', 'mixed'
    coverage: float  # Total pixel coverage
    lab_range: dict[str, tuple[float, float]]  # {'L': (min, max), ...}
    family_span: list[int]  # Which family indices this gradient spans


@dataclass
class ColorMetrics:
    """Per-color metrics."""
    coverage: float
    chroma: float
    hue: float
    local_contrast: float
    coherence: float
    isolation: float
    lab: np.ndarray


@dataclass
class FeatureData:
    """Output of Stage 2: Feature Extraction."""
    stability: dict[BinKey, StabilityInfo]
    families: list[ColorFamily]
    gradients: list[GradientChain]
    metrics: dict[BinKey, ColorMetrics]
    adjacency: dict[tuple[BinKey, BinKey], int]  # (bin1, bin2) -> count


def analyze_scale_stability(data: PreparedData) -> dict:
    """
    Stage 2a: Classify each coarse bin by its fine-scale structure.

    ANCHOR: 1-2 fine children, color is stable
    GRADIENT: 3+ fine children with monotonic LAB progression
    TEXTURE: 3+ fine children scattered in LAB space
    """
    stability = {}
    fine_size = FINE_SCALE * JND

    for coarse_bin, fine_list in data.coarse_to_fine.items():
        fine_count = len(fine_list)

        if fine_count <= 2:
            stability[coarse_bin] = StabilityInfo(
                stability_type='anchor',
                fine_count=fine_count,
                fine_variance=0.0
            )
            continue

        # Get LAB values of fine children
        fine_labs = np.array([np.array(fb) * fine_size for fb in fine_list])

        # Compute variance in each dimension
        var_L = np.var(fine_labs[:, 0])
        var_a = np.var(fine_labs[:, 1])
        var_b = np.var(fine_labs[:, 2])
        total_var = var_L + var_a + var_b

        # Check for monotonic progression (gradient vs texture)
        # A gradient has most variance in one axis with monotonic progression
        max_var_axis = np.argmax([var_L, var_a, var_b])
        axis_names = ['L', 'a', 'b']

        # Sort fine bins by the dominant axis
        sorted_labs = fine_labs[fine_labs[:, max_var_axis].argsort()]

        # Check monotonicity: differences should be mostly same sign
        diffs = np.diff(sorted_labs[:, max_var_axis])
        if len(diffs) > 0:
            pos_ratio = np.sum(diffs >= 0) / len(diffs)
            is_monotonic = pos_ratio > 0.7 or pos_ratio < 0.3
        else:
            is_monotonic = False

        # Gradient: one dominant axis with monotonic progression
        dominant_var_ratio = max(var_L, var_a, var_b) / (total_var + 1e-10)

        if is_monotonic and dominant_var_ratio > 0.5:
            stability[coarse_bin] = StabilityInfo(
                stability_type='gradient',
                fine_count=fine_count,
                fine_variance=total_var,
                dominant_axis=axis_names[max_var_axis]
            )
        else:
            stability[coarse_bin] = StabilityInfo(
                stability_type='texture',
                fine_count=fine_count,
                fine_variance=total_var
            )

    return stability


def detect_color_families(data: PreparedData, stability: dict) -> list:
    """
    Stage 2b: Group coarse bins by hue similarity.
    """
    coarse_size = COARSE_SCALE * JND
    families = []
    assigned = set()

    # Compute hue and chroma for each coarse bin
    bin_info = {}
    for coarse_bin, bin_data in data.coarse_bins.items():
        lab = bin_data.lab
        chroma = compute_chroma(lab)
        hue = compute_hue(lab)
        coverage = bin_data.count / data.total_pixels
        bin_info[coarse_bin] = {
            'lab': lab,
            'chroma': chroma,
            'hue': hue,
            'coverage': coverage,
            'L': lab[0]
        }

    # Adaptive chroma threshold based on distribution
    chromas = [info['chroma'] for info in bin_info.values()]
    median_chroma = np.median(chromas) if chromas else 0
    chroma_threshold = max(10, median_chroma * 0.5)

    # Separate chromatic and neutral
    chromatic_bins = {b: info for b, info in bin_info.items() if info['chroma'] >= chroma_threshold}
    neutral_bins = {b: info for b, info in bin_info.items() if info['chroma'] < chroma_threshold}

    # Sort by coverage to seed clusters with most significant colors
    sorted_chromatic = sorted(chromatic_bins.items(), key=lambda x: -x[1]['coverage'])

    for seed_bin, seed_info in sorted_chromatic:
        if seed_bin in assigned:
            continue

        # Find all bins within hue range
        cluster_members = [seed_bin]
        assigned.add(seed_bin)

        for other_bin, other_info in chromatic_bins.items():
            if other_bin in assigned:
                continue

            hue_diff = circular_hue_distance(seed_info['hue'], other_info['hue'])

            if hue_diff <= HUE_CLUSTER_RANGE:
                cluster_members.append(other_bin)
                assigned.add(other_bin)

        # Compute family properties
        member_infos = [bin_info[b] for b in cluster_members]
        hues = [info['hue'] for info in member_infos]
        lightnesses = [info['L'] for info in member_infos]
        chromas_cluster = [info['chroma'] for info in member_infos]
        coverages = [info['coverage'] for info in member_infos]

        # Circular mean for hue
        sin_sum = sum(math.sin(math.radians(h)) for h in hues)
        cos_sum = sum(math.cos(math.radians(h)) for h in hues)
        hue_center = math.degrees(math.atan2(sin_sum, cos_sum)) % 360

        families.append(ColorFamily(
            hue_center=hue_center,
            lightness_range=(min(lightnesses), max(lightnesses)),
            chroma_range=(min(chromas_cluster), max(chromas_cluster)),
            total_coverage=sum(coverages),
            members=cluster_members,
            is_neutral=False
        ))

    # Add neutral family if significant
    if neutral_bins:
        neutral_members = list(neutral_bins.keys())
        neutral_infos = [bin_info[b] for b in neutral_members]
        lightnesses = [info['L'] for info in neutral_infos]
        coverages = [info['coverage'] for info in neutral_infos]
        chromas_neutral = [info['chroma'] for info in neutral_infos]

        families.append(ColorFamily(
            hue_center=0,  # N/A for neutral
            lightness_range=(min(lightnesses), max(lightnesses)),
            chroma_range=(min(chromas_neutral), max(chromas_neutral)),
            total_coverage=sum(coverages),
            members=neutral_members,
            is_neutral=True
        ))

    return families


def build_adjacency_graph(data: PreparedData) -> tuple[dict, dict]:
    """
    Build adjacency graph at fine scale with 8-connectivity.

    Uses vectorized numpy operations with bin encoding for fast aggregation.

    Returns:
        adjacency: {(b1, b2): count} - symmetric edge counts
        directional: {(b1, b2): {'right': n, 'left': n, ...}} - directional counts
    """
    fine_binned = data.fine_binned  # (h, w, 3) array

    # Direction mappings: (dy, dx) -> direction name
    DIRECTIONS = [
        ((-1, -1), 'up_left'),    ((-1, 0), 'above'),    ((-1, 1), 'up_right'),
        ((0, -1), 'left'),                                ((0, 1), 'right'),
        ((1, -1), 'down_left'),   ((1, 0), 'below'),     ((1, 1), 'down_right')
    ]
    ALL_DIRS = [name for _, name in DIRECTIONS]

    # Collect all edges across all directions
    all_src_encoded: list[np.ndarray] = []
    all_nbr_encoded: list[np.ndarray] = []
    all_dir_indices: list[np.ndarray] = []

    for dir_idx, ((dy, dx), dir_name) in enumerate(DIRECTIONS):
        # Define slices for source and neighbor regions
        if dy == -1:
            src_y, nbr_y = slice(1, None), slice(None, -1)
        elif dy == 1:
            src_y, nbr_y = slice(None, -1), slice(1, None)
        else:
            src_y, nbr_y = slice(None), slice(None)

        if dx == -1:
            src_x, nbr_x = slice(1, None), slice(None, -1)
        elif dx == 1:
            src_x, nbr_x = slice(None, -1), slice(1, None)
        else:
            src_x, nbr_x = slice(None), slice(None)

        # Extract aligned regions
        src = fine_binned[src_y, src_x]  # Source pixels
        nbr = fine_binned[nbr_y, nbr_x]  # Neighbor pixels in this direction

        # Find where colors differ (edge exists)
        diff_mask = np.any(src != nbr, axis=2)

        # Get bin values at edge locations
        edge_ys, edge_xs = np.where(diff_mask)
        if len(edge_ys) == 0:
            continue

        src_bins = src[edge_ys, edge_xs]  # (n_edges, 3)
        nbr_bins = nbr[edge_ys, edge_xs]  # (n_edges, 3)

        # Encode bins to int64 for fast operations
        src_encoded = encode_bins(src_bins)
        nbr_encoded = encode_bins(nbr_bins)

        all_src_encoded.append(src_encoded)
        all_nbr_encoded.append(nbr_encoded)
        all_dir_indices.append(np.full(len(src_encoded), dir_idx, dtype=np.int64))

    # Early exit if no edges found
    if not all_src_encoded:
        return {}, {}

    # Concatenate all edges
    src_all = np.concatenate(all_src_encoded)
    nbr_all = np.concatenate(all_nbr_encoded)
    dir_all = np.concatenate(all_dir_indices)

    # === Build symmetric adjacency using np.unique ===
    # Sort each pair so (a,b) and (b,a) map to same key
    pairs = np.column_stack([src_all, nbr_all])
    symmetric_pairs = np.sort(pairs, axis=1)

    # Aggregate counts with np.unique
    unique_sym, sym_counts = np.unique(symmetric_pairs, axis=0, return_counts=True)

    # Decode all bins at once (vectorized), then build dict
    decoded_sym_b1 = decode_bins(unique_sym[:, 0])  # (n, 3)
    decoded_sym_b2 = decode_bins(unique_sym[:, 1])  # (n, 3)

    adjacency: dict[tuple[BinKey, BinKey], int] = {}
    for i in range(len(unique_sym)):
        b1 = tuple(decoded_sym_b1[i])
        b2 = tuple(decoded_sym_b2[i])
        adjacency[(b1, b2)] = int(sym_counts[i])

    # === Build directional counts using np.unique ===
    # Stack (src, nbr, direction) triplets
    directed_triplets = np.column_stack([src_all, nbr_all, dir_all])
    unique_dir, dir_counts = np.unique(directed_triplets, axis=0, return_counts=True)

    # Decode all bins at once (vectorized), then build dict
    decoded_dir_b1 = decode_bins(unique_dir[:, 0])  # (n, 3)
    decoded_dir_b2 = decode_bins(unique_dir[:, 1])  # (n, 3)

    directional: dict[tuple[BinKey, BinKey], dict[str, int]] = {}
    for i in range(len(unique_dir)):
        b1 = tuple(decoded_dir_b1[i])
        b2 = tuple(decoded_dir_b2[i])
        dir_idx = int(unique_dir[i, 2])
        dir_name = ALL_DIRS[dir_idx]

        key = (b1, b2)
        if key not in directional:
            directional[key] = {d: 0 for d in ALL_DIRS}
        directional[key][dir_name] = int(dir_counts[i])

    return adjacency, directional


def compute_color_metrics(data: PreparedData, adjacency: dict, stability: dict) -> dict:
    """
    Stage 2c: Compute per-color metrics for coarse bins.
    """
    coarse_size = COARSE_SCALE * JND
    fine_size = FINE_SCALE * JND
    metrics = {}

    # Build neighbor lookup for coarse bins (via fine adjacency)
    coarse_neighbors = {b: set() for b in data.coarse_bins}
    coarse_adjacency_count = {b: 0 for b in data.coarse_bins}

    for (f1, f2), count in adjacency.items():
        c1 = data.fine_to_coarse.get(f1)
        c2 = data.fine_to_coarse.get(f2)
        if c1 and c2 and c1 != c2:
            coarse_neighbors[c1].add(c2)
            coarse_neighbors[c2].add(c1)
            coarse_adjacency_count[c1] += count
            coarse_adjacency_count[c2] += count

    # Compute local contrast (average LAB distance to neighbors)
    local_contrast = {}
    for coarse_bin, neighbors in coarse_neighbors.items():
        if not neighbors:
            local_contrast[coarse_bin] = 0.0
            continue

        lab = data.coarse_bins[coarse_bin].lab
        distances = []
        for neighbor in neighbors:
            neighbor_lab = data.coarse_bins[neighbor].lab
            distances.append(np.linalg.norm(lab - neighbor_lab))
        local_contrast[coarse_bin] = np.mean(distances) if distances else 0.0

    # Compute coherence (largest blob / total pixels)
    coherence = compute_coherence(data)

    # Compute isolation (inverse of neighbor count, normalized)
    max_neighbors = max(len(n) for n in coarse_neighbors.values()) if coarse_neighbors else 1

    for coarse_bin, bin_data in data.coarse_bins.items():
        lab = bin_data.lab
        chroma = compute_chroma(lab)
        hue = compute_hue(lab)
        coverage = bin_data.count / data.total_pixels

        neighbor_count = len(coarse_neighbors[coarse_bin])
        isolation = 1.0 - (neighbor_count / (max_neighbors + 1))

        metrics[coarse_bin] = ColorMetrics(
            coverage=coverage,
            chroma=chroma,
            hue=hue,
            local_contrast=local_contrast.get(coarse_bin, 0.0),
            coherence=coherence.get(coarse_bin, 0.0),
            isolation=isolation,
            lab=lab
        )

    return metrics


def compute_coherence(data: PreparedData) -> dict[BinKey, float]:
    """Compute spatial coherence for each coarse bin."""
    coarse_binned = data.coarse_binned
    h, w = data.image_shape

    # Pre-encode coarse bins to int64 for fast scalar comparison
    # This replaces 3-element array comparison with single integer comparison
    coarse_encoded = encode_bins(coarse_binned.reshape(-1, 3)).reshape(h, w)

    coherence: dict[BinKey, float] = {}
    structure = np.ones((3, 3), dtype=int)

    for coarse_bin, bin_data in data.coarse_bins.items():
        if bin_data.count == 0:
            coherence[coarse_bin] = 0.0
            continue

        # Encode bin key and compare as scalar (much faster than 3-way comparison)
        bin_encoded = encode_bin_key(coarse_bin)
        mask = coarse_encoded == bin_encoded

        # Find connected components
        labeled, num_blobs = label(mask, structure=structure)

        if num_blobs == 0:
            coherence[coarse_bin] = 0.0
            continue

        blob_sizes = np.bincount(labeled.ravel())[1:]
        largest_blob = int(np.max(blob_sizes))
        coherence[coarse_bin] = largest_blob / bin_data.count

    return coherence


def compute_fine_coherence(data: PreparedData, coarse_coherence: dict) -> dict:
    """
    Map fine bins to their parent coarse bin's coherence.

    Computing coherence per fine bin is expensive (O(n) label operations).
    Instead, use the coarse bin's coherence as a proxy — fine bins inherit
    their parent's coherence score.
    """
    coherence = {}
    for fine_bin in data.fine_bins:
        coarse = data.fine_to_coarse.get(fine_bin)
        if coarse:
            coherence[fine_bin] = coarse_coherence.get(coarse, 0.0)
        else:
            coherence[fine_bin] = 0.0
    return coherence


def analyze_gradient_flow(directional: dict, min_total: int = 50) -> dict:
    """
    Analyze directional adjacency to find gradient flow patterns.

    For each color pair, computes:
    - h_flow: right - left (positive = flows right)
    - v_flow: below - above (positive = flows down)
    - Asymmetry: how one-directional the flow is (0-1)

    Args:
        directional: {(b1, b2): {'right': n, 'left': n, ...}}
        min_total: minimum total edge count to include pair

    Returns:
        {(b1, b2): {'h_flow': n, 'v_flow': n, 'h_asymmetry': f, ...}}
    """
    flow_analysis = {}

    for (b1, b2), dirs in directional.items():
        total = sum(dirs.values())
        if total < min_total:
            continue

        # Horizontal flow (positive = b2 is right of b1)
        h_flow = dirs['right'] - dirs['left']
        # Vertical flow (positive = b2 is below b1)
        v_flow = dirs['below'] - dirs['above']
        # Diagonal flows
        dr_flow = dirs['down_right'] - dirs['up_left']
        dl_flow = dirs['down_left'] - dirs['up_right']

        # Compute asymmetry (how one-directional)
        h_total = dirs['right'] + dirs['left']
        v_total = dirs['below'] + dirs['above']
        dr_total = dirs['down_right'] + dirs['up_left']
        dl_total = dirs['down_left'] + dirs['up_right']

        h_asymmetry = abs(h_flow) / (h_total + 1) if h_total > 0 else 0
        v_asymmetry = abs(v_flow) / (v_total + 1) if v_total > 0 else 0
        dr_asymmetry = abs(dr_flow) / (dr_total + 1) if dr_total > 0 else 0
        dl_asymmetry = abs(dl_flow) / (dl_total + 1) if dl_total > 0 else 0

        flow_analysis[(b1, b2)] = {
            'h_flow': h_flow,
            'v_flow': v_flow,
            'dr_flow': dr_flow,
            'dl_flow': dl_flow,
            'h_asymmetry': h_asymmetry,
            'v_asymmetry': v_asymmetry,
            'dr_asymmetry': dr_asymmetry,
            'dl_asymmetry': dl_asymmetry,
            'total': total
        }

    return flow_analysis


def compute_local_angles(fine_members: list) -> list:
    """
    Compute local angle at each interior point in the gradient chain.

    At each point (except endpoints), compute the angle between:
    - Incoming direction (from previous point)
    - Outgoing direction (to next point)

    Returns list of angles in degrees. Empty list if chain too short.
    """
    if len(fine_members) < 3:
        return []

    fine_size = FINE_SCALE * JND

    # Convert fine bins to LAB coordinates
    labs = [np.array(fb) * fine_size for fb in fine_members]

    angles = []
    for i in range(1, len(labs) - 1):
        # Incoming vector: from previous to current
        v_in = labs[i] - labs[i-1]
        # Outgoing vector: from current to next
        v_out = labs[i+1] - labs[i]

        # Compute angle between vectors
        norm_in = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)

        if norm_in < 1e-6 or norm_out < 1e-6:
            # Zero-length vector (duplicate point), skip — can't determine direction
            continue

        # Cosine of angle
        cos_angle = np.dot(v_in, v_out) / (norm_in * norm_out)
        # Clamp to [-1, 1] to handle numerical issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Convert to degrees
        angle_deg = np.degrees(np.arccos(cos_angle))
        angles.append(angle_deg)

    return angles


def detect_flow_gradients(data: PreparedData, directional: dict,
                          fine_coherence: dict, families: list,
                          min_chain_length: int = 3,
                          min_asymmetry: float = 0.3) -> list:
    """
    Detect gradients by following directional flow through the adjacency graph.

    A gradient is a chain where colors consistently flow in one spatial direction:
    - Horizontal: each color has the next color predominantly to its right (or left)
    - Vertical: each color has the next color predominantly below (or above)
    - Diagonal: flows along diagonals

    This approach can capture multi-hue gradients (e.g., red→orange→yellow)
    as long as they're spatially smooth.

    Args:
        data: Prepared image data
        directional: Directional adjacency from build_adjacency_graph
        fine_coherence: Per-fine-bin coherence scores
        families: Color families for span detection
        min_chain_length: Minimum colors to form a gradient
        min_asymmetry: Minimum flow asymmetry to follow an edge

    Returns:
        List of GradientChain objects
    """
    fine_size = FINE_SCALE * JND
    colors = list(data.fine_bins.keys())

    # Analyze flow patterns
    # Lower threshold because fine bins have fewer pixels per edge than coarse bins
    flow = analyze_gradient_flow(directional, min_total=10)

    # Build directed flow graph
    # Edge from A to B if B is predominantly in one direction from A
    ALL_DIRECTIONS = ['right', 'left', 'below', 'above',
                      'down_right', 'down_left', 'up_right', 'up_left']
    flow_graph = {b: {d: [] for d in ALL_DIRECTIONS} for b in colors}

    def get_coherence(b):
        return fine_coherence.get(b, 1.0)

    for (b1, b2), analysis in flow.items():
        if b1 not in flow_graph or b2 not in flow_graph:
            continue

        # Weight by geometric mean of coherences
        coh1, coh2 = get_coherence(b1), get_coherence(b2)
        coherence_weight = np.sqrt(coh1 * coh2)

        # Check horizontal flow
        if analysis['h_asymmetry'] > min_asymmetry:
            if analysis['h_flow'] > 0:
                weighted = analysis['h_flow'] * coherence_weight
                flow_graph[b1]['right'].append((b2, weighted, analysis['total']))
            else:
                weighted = -analysis['h_flow'] * coherence_weight
                flow_graph[b1]['left'].append((b2, weighted, analysis['total']))

        # Check vertical flow
        if analysis['v_asymmetry'] > min_asymmetry:
            if analysis['v_flow'] > 0:
                weighted = analysis['v_flow'] * coherence_weight
                flow_graph[b1]['below'].append((b2, weighted, analysis['total']))
            else:
                weighted = -analysis['v_flow'] * coherence_weight
                flow_graph[b1]['above'].append((b2, weighted, analysis['total']))

        # Check diagonal flows
        if analysis['dr_asymmetry'] > min_asymmetry:
            if analysis['dr_flow'] > 0:
                weighted = analysis['dr_flow'] * coherence_weight
                flow_graph[b1]['down_right'].append((b2, weighted, analysis['total']))
            else:
                weighted = -analysis['dr_flow'] * coherence_weight
                flow_graph[b1]['up_left'].append((b2, weighted, analysis['total']))

        if analysis['dl_asymmetry'] > min_asymmetry:
            if analysis['dl_flow'] > 0:
                weighted = analysis['dl_flow'] * coherence_weight
                flow_graph[b1]['down_left'].append((b2, weighted, analysis['total']))
            else:
                weighted = -analysis['dl_flow'] * coherence_weight
                flow_graph[b1]['up_right'].append((b2, weighted, analysis['total']))

    # Sort neighbors by coherence-weighted flow strength
    for b in flow_graph:
        for direction in flow_graph[b]:
            flow_graph[b][direction].sort(key=lambda x: x[1], reverse=True)

    def follow_flow(start, direction):
        """Follow flow in one direction to build a gradient chain."""
        chain = [start]
        current = start
        visited = {start}

        while True:
            candidates = flow_graph[current][direction]
            next_color = None
            for neighbor, strength, total in candidates:
                if neighbor not in visited:
                    next_color = neighbor
                    break

            if next_color is None:
                break

            chain.append(next_color)
            visited.add(next_color)
            current = next_color

        return chain

    # Compute seed scores combining multiple factors
    total_pixels = data.total_pixels

    # Find LAB extremes for bonus scoring
    all_labs = {b: np.array(b) * fine_size for b in colors}
    l_values = {b: lab[0] for b, lab in all_labs.items()}
    a_values = {b: lab[1] for b, lab in all_labs.items()}
    b_values = {b: lab[2] for b, lab in all_labs.items()}

    l_min_bin = min(l_values, key=l_values.get)
    l_max_bin = max(l_values, key=l_values.get)
    a_min_bin = min(a_values, key=a_values.get)
    a_max_bin = max(a_values, key=a_values.get)
    b_min_bin = min(b_values, key=b_values.get)
    b_max_bin = max(b_values, key=b_values.get)
    extreme_bins = {l_min_bin, l_max_bin, a_min_bin, a_max_bin, b_min_bin, b_max_bin}

    # Compute local contrast for each color (avg LAB distance to flow neighbors)
    local_contrast = {}
    for b in colors:
        neighbors = set()
        for direction in ALL_DIRECTIONS:
            for neighbor, _, _ in flow_graph[b][direction]:
                neighbors.add(neighbor)
        if neighbors:
            b_lab = all_labs[b]
            distances = [np.linalg.norm(b_lab - all_labs[n]) for n in neighbors if n in all_labs]
            local_contrast[b] = np.mean(distances) if distances else 0
        else:
            local_contrast[b] = 0

    # Normalize contrast (typically 0-30 LAB units range)
    max_contrast = max(local_contrast.values()) if local_contrast else 1
    contrast_normalized = {b: c / max_contrast for b, c in local_contrast.items()}

    def compute_seed_score(b):
        # Coverage: scale so 10% coverage ≈ 1.0
        cov = data.fine_bins[b].count / total_pixels
        cov_score = cov * 10

        # Contrast boost: up to 0.3 for high contrast colors
        contrast_score = contrast_normalized.get(b, 0) * 0.3

        # LAB extreme boost: bonus for colors at extremes
        extreme_score = 0.2 if b in extreme_bins else 0

        # Coherence weight: reduce score for scattered/noisy colors
        coherence = fine_coherence.get(b, 0.5)

        return (cov_score + contrast_score + extreme_score) * coherence

    scored_colors = [(b, compute_seed_score(b)) for b in colors]
    scored_colors.sort(key=lambda x: x[1], reverse=True)

    all_gradients = []

    # Try all directions from top-scored colors
    for direction in ALL_DIRECTIONS:
        for start, _ in scored_colors[:MAX_GRADIENT_SEARCH_SEEDS]:
            chain = follow_flow(start, direction)

            if len(chain) >= min_chain_length:
                chain_coverage = sum(data.fine_bins[b].count for b in chain) / total_pixels

                # Compute LAB bounds
                labs = [np.array(b) * fine_size for b in chain]
                lab_array = np.array(labs)
                l_min, l_max = lab_array[:, 0].min(), lab_array[:, 0].max()
                a_min, a_max = lab_array[:, 1].min(), lab_array[:, 1].max()
                b_min, b_max = lab_array[:, 2].min(), lab_array[:, 2].max()

                all_gradients.append({
                    'chain': chain,
                    'direction': direction,
                    'coverage': chain_coverage,
                    'l_range': l_max - l_min,
                    'a_range': a_max - a_min,
                    'b_range': b_max - b_min,
                    'l_bounds': (l_min, l_max),
                    'a_bounds': (a_min, a_max),
                    'b_bounds': (b_min, b_max),
                    'score': len(chain) * chain_coverage
                })

    # Sort by score
    all_gradients.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate by LAB bounds overlap
    def bounds_overlap(bounds1, bounds2):
        """Compute overlap ratio: how much of bounds1 is inside bounds2."""
        min1, max1 = bounds1
        min2, max2 = bounds2
        range1 = max1 - min1
        if range1 < 0.1:
            return 1.0 if min2 <= min1 <= max2 else 0.0
        overlap_min = max(min1, min2)
        overlap_max = min(max1, max2)
        overlap = max(0, overlap_max - overlap_min)
        return overlap / range1

    final_gradients = []
    for grad in all_gradients:
        is_redundant = False
        for existing in final_gradients:
            l_overlap = bounds_overlap(grad['l_bounds'], existing['l_bounds'])
            a_overlap = bounds_overlap(grad['a_bounds'], existing['a_bounds'])
            b_overlap = bounds_overlap(grad['b_bounds'], existing['b_bounds'])
            if l_overlap >= 0.5 and a_overlap >= 0.5 and b_overlap >= 0.5:
                is_redundant = True
                break
        if not is_redundant:
            final_gradients.append(grad)

    # Convert to GradientChain objects
    result = []
    for grad in final_gradients[:MAX_NOTABLE_COLORS]:
        # Map spatial direction to display direction
        dir_name = grad['direction']
        if dir_name in ('right', 'left'):
            display_dir = 'horizontal'
        elif dir_name in ('below', 'above'):
            display_dir = 'vertical'
        elif dir_name in ('down_right', 'up_left'):
            display_dir = 'diagonal'
        else:
            display_dir = 'anti-diagonal'

        # Map fine bins to coarse bins for stops
        coarse_stops = []
        seen_coarse = set()
        for fine_bin in grad['chain']:
            coarse = data.fine_to_coarse.get(fine_bin)
            if coarse and coarse not in seen_coarse:
                coarse_stops.append(coarse)
                seen_coarse.add(coarse)

        if len(coarse_stops) < 2:
            continue

        # Filter out low-variation gradients (look like solid blocks)
        l_span = grad['l_bounds'][1] - grad['l_bounds'][0]
        a_span = grad['a_bounds'][1] - grad['a_bounds'][0]
        b_span = grad['b_bounds'][1] - grad['b_bounds'][0]
        total_span = l_span + a_span + b_span
        if total_span < 40:  # ~17 JND minimum variation
            continue

        # Filter out directionally incoherent gradients (noisy zig-zags)
        # High mean angle indicates random direction changes rather than smooth
        # progression through LAB space (90° = orthogonal = random walk)
        angles = compute_local_angles(grad['chain'])
        if angles and np.mean(angles) >= GRADIENT_ANGLE_THRESHOLD:
            continue

        # Determine which color families this gradient spans
        family_indices = set()
        for stop in coarse_stops:
            lab = data.coarse_bins[stop].lab
            hue = compute_hue(lab)
            chroma = compute_chroma(lab)
            # Skip neutral colors (low chroma)
            if chroma < 8:
                continue
            for i, family in enumerate(families):
                if family.is_neutral:
                    continue
                if circular_hue_distance(hue, family.hue_center) <= HUE_CLUSTER_RANGE:
                    family_indices.add(i)
                    break

        result.append(GradientChain(
            stops=coarse_stops,
            fine_members=grad['chain'],
            direction=display_dir,
            coverage=grad['coverage'],
            lab_range={
                'L': grad['l_bounds'],
                'a': grad['a_bounds'],
                'b': grad['b_bounds']
            },
            family_span=list(family_indices)
        ))

    return result


def extract_features(data: PreparedData) -> FeatureData:
    """Stage 2: Extract all features from prepared data."""
    stability = analyze_scale_stability(data)
    families = detect_color_families(data, stability)
    adjacency, directional = build_adjacency_graph(data)
    metrics = compute_color_metrics(data, adjacency, stability)

    # Extract coarse coherence from metrics, then map to fine bins
    coarse_coherence = {b: m.coherence for b, m in metrics.items()}
    fine_coherence = compute_fine_coherence(data, coarse_coherence)

    gradients = detect_flow_gradients(data, directional, fine_coherence, families)

    return FeatureData(
        stability=stability,
        families=families,
        gradients=gradients,
        metrics=metrics,
        adjacency=adjacency
    )


# =============================================================================
# Stage 3: Synthesis
# =============================================================================

@dataclass
class NotableColor:
    """A significant color in the palette."""
    coarse_bin: BinKey
    lab: np.ndarray
    hex: str
    rgb: tuple[int, int, int]
    name: str
    role: str  # 'dominant', 'secondary', 'accent', 'dark', 'light'
    coverage: float
    chroma: float
    significance: float
    characteristics: list[str]
    gradient_membership: list[int]  # Indices into gradients list
    family_index: int = 0  # Which perceptual family this color belongs to


@dataclass
class ContrastPair:
    """A pair of colors with high contrast."""
    color_a: str  # Color name
    color_b: str
    delta_l: float
    contrast_ratio: float
    wcag_level: str  # 'AAA', 'AA', 'AA-large', 'fail'


@dataclass
class HarmonicPair:
    """A pair of colors with similar hue."""
    color_a: str
    color_b: str
    hue_difference: float


@dataclass
class HueStructure:
    """Factual hue data derived from displayed families."""
    chromatic_count: int  # Number of chromatic (non-neutral) families
    hue_centers: list[float]  # Hue angles (0-360) for each chromatic family
    separations: list[float]  # Angular distances between adjacent hues (sorted)


@dataclass
class SynthesisResult:
    """Output of Stage 3: Synthesis."""
    hue_structure: HueStructure
    notable_colors: list[NotableColor]
    gradients: list[GradientChain]
    contrast_pairs: list[ContrastPair]
    harmonic_pairs: list[HarmonicPair]
    distribution_analysis: str
    lightness_range: tuple[float, float]
    chroma_range: tuple[float, float]
    family_count: int = 0  # Number of distinct color families in palette


# =============================================================================
# Family-based Color Clustering
# =============================================================================

def cluster_into_families(metrics: dict[BinKey, ColorMetrics],
                          threshold: float = FAMILY_CLUSTER_THRESHOLD) -> list[list[BinKey]]:
    """
    Cluster coarse bins into color families using hierarchical clustering.

    Colors within `threshold` LAB distance are grouped together, capturing
    perceptually similar colors regardless of their hue/lightness classification.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    bins = list(metrics.keys())
    if len(bins) < 2:
        return [[b] for b in bins]

    # Build LAB matrix from metrics
    labs = np.array([metrics[b].lab for b in bins])

    # Hierarchical clustering with average linkage
    Z = linkage(labs, method='average', metric='euclidean')
    labels = fcluster(Z, t=threshold, criterion='distance')

    # Group bins by cluster label
    families: dict[int, list[BinKey]] = {}
    for bin_key, label in zip(bins, labels):
        families.setdefault(label, []).append(bin_key)

    return list(families.values())


@dataclass
class FamilyCluster:
    """A cluster of perceptually similar colors (LAB-based)."""
    bins: list[BinKey]
    coverage: float  # Total coverage of all bins in family
    avg_lab: np.ndarray  # Centroid in LAB space
    avg_chroma: float
    centroid_bin: BinKey  # Bin closest to centroid
    family_type: str  # 'chromatic', 'dark', 'light', 'neutral_mid'


def analyze_families(metrics: dict[BinKey, ColorMetrics],
                     families: list[list[BinKey]]) -> list[FamilyCluster]:
    """Compute aggregate statistics for each color family."""
    family_stats = []

    for family_bins in families:
        total_coverage = sum(metrics[b].coverage for b in family_bins)
        avg_lab = np.mean([metrics[b].lab for b in family_bins], axis=0)
        avg_chroma = np.mean([metrics[b].chroma for b in family_bins])
        avg_L = avg_lab[0]

        # Find the bin closest to the family centroid
        centroid_bin = min(family_bins,
                           key=lambda b: np.linalg.norm(metrics[b].lab - avg_lab))

        # Classify family type
        if avg_chroma < 15:
            if avg_L < 25:
                family_type = 'dark'
            elif avg_L > 75:
                family_type = 'light'
            else:
                family_type = 'neutral_mid'
        else:
            family_type = 'chromatic'

        family_stats.append(FamilyCluster(
            bins=family_bins,
            coverage=total_coverage,
            avg_lab=avg_lab,
            avg_chroma=avg_chroma,
            centroid_bin=centroid_bin,
            family_type=family_type
        ))

    # Sort by coverage descending
    family_stats.sort(key=lambda f: -f.coverage)
    return family_stats


def select_colors_by_family(metrics: dict[BinKey, ColorMetrics],
                            families: list[FamilyCluster],
                            max_colors: int = MAX_NOTABLE_COLORS) -> list[tuple[BinKey, str]]:
    """
    Select notable colors using family-first approach.

    1. Every family above coverage threshold gets one representative
    2. High-chroma families get representation even at low coverage (accents)
    3. Fill remaining slots with additional shades from largest families

    Returns list of (bin_key, role) tuples.
    """
    selected: list[tuple[BinKey, str]] = []
    selected_bins: set[BinKey] = set()

    # Step 1: One representative per significant family (by coverage order)
    significant_families = [f for f in families if f.coverage >= MIN_FAMILY_COVERAGE]

    for fam in significant_families:
        if len(selected) >= max_colors:
            break
        # Pick the bin with highest coverage in this family
        best_bin = max(fam.bins, key=lambda b: metrics[b].coverage)
        role = fam.family_type
        selected.append((best_bin, role))
        selected_bins.add(best_bin)

    # Step 2: High-chroma accent families (even if low coverage)
    accent_families = [f for f in families
                       if f.coverage < MIN_FAMILY_COVERAGE
                       and f.avg_chroma > ACCENT_CHROMA_THRESHOLD]

    for fam in accent_families:
        if len(selected) >= max_colors:
            break
        # For accents, pick the most saturated bin
        best_bin = max(fam.bins, key=lambda b: metrics[b].chroma)
        if best_bin not in selected_bins:
            selected.append((best_bin, 'accent'))
            selected_bins.add(best_bin)

    # Step 3: Fill remaining slots with additional shades from largest families
    for fam in families:
        if len(selected) >= max_colors:
            break

        available = [b for b in fam.bins if b not in selected_bins]
        available.sort(key=lambda b: -metrics[b].coverage)

        for b in available[:2]:  # Up to 2 more per family
            if len(selected) >= max_colors:
                break
            if metrics[b].coverage > 0.005:  # Minimum 0.5% coverage for fills
                selected.append((b, 'fill'))
                selected_bins.add(b)

    return selected


def compute_significance(metrics: ColorMetrics, stability: StabilityInfo,
                         median_chroma: float, chroma_iqr: float) -> float:
    """Compute significance score for a color.

    Coverage is the primary factor. Chroma only provides a small bonus
    for colors that stand out from a muted background - it shouldn't
    override high coverage.
    """
    # Base: coverage (0-100 scale) - this is the primary factor
    score = metrics.coverage * 100

    # Chroma bonus: modest bonus for standing out in muted image
    # Capped to avoid overwhelming coverage
    if chroma_iqr > 0 and metrics.chroma > median_chroma:
        chroma_bonus = min(5, (metrics.chroma - median_chroma) / (chroma_iqr + 1) * 2)
        score += chroma_bonus

    # Isolation bonus: only significant if has some coverage
    if metrics.coverage > 0.001:
        score += metrics.isolation * 3

    # Coherence bonus: forms a blob, not noise
    score += metrics.coherence * 5

    # Stability bonus
    if stability.stability_type == 'anchor':
        score += 3
    elif stability.stability_type == 'gradient':
        score += 1

    return score


def generate_color_name(lab: np.ndarray) -> str:
    """Find the nearest XKCD color name for a LAB coordinate."""
    names, labs = _get_xkcd_lab_colors()
    # Squared Euclidean distance (sqrt unnecessary for argmin)
    distances = np.sum((labs - lab) ** 2, axis=1)
    return names[np.argmin(distances)]


def classify_role(metrics: ColorMetrics, coverage_rank: int, is_accent: bool) -> str:
    """Classify the descriptive role of a color based on pre-computed rank."""
    if is_accent:
        return "accent"

    if coverage_rank == 0:
        return "dominant"
    elif coverage_rank <= 2:
        return "secondary"
    elif metrics.lab[0] < 30:
        return "dark"
    elif metrics.lab[0] > 70:
        return "light"
    else:
        return "secondary"


def generate_characteristics(metrics: ColorMetrics, stability: StabilityInfo,
                             role: str, gradient_count: int) -> list:
    """Generate descriptive characteristics for a color."""
    chars = []

    if role == "dominant":
        chars.append("Largest coverage, anchors the palette")
    elif role == "accent":
        chars.append("High chroma focal point")
        if metrics.isolation > 0.7:
            chars.append("spatially isolated")

    if stability.stability_type == 'anchor':
        chars.append("Stable across scales")
    elif stability.stability_type == 'texture':
        chars.append("Textured/varied region")

    if metrics.coherence > 0.7:
        chars.append("Forms coherent region")
    elif metrics.coherence < 0.3:
        chars.append("Scattered distribution")

    if gradient_count > 0:
        chars.append(f"Part of {gradient_count} gradient(s)")

    return chars


def compute_wcag_contrast(lab1: np.ndarray, lab2: np.ndarray) -> tuple:
    """
    Compute WCAG contrast ratio between two colors.
    Returns (ratio, level).
    """
    # Convert to relative luminance via RGB
    rgb1 = lab_to_rgb(lab1.reshape(1, -1))[0] / 255.0
    rgb2 = lab_to_rgb(lab2.reshape(1, -1))[0] / 255.0

    def relative_luminance(rgb):
        r, g, b = rgb
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    l1 = relative_luminance(rgb1)
    l2 = relative_luminance(rgb2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    ratio = (lighter + 0.05) / (darker + 0.05)

    if ratio >= 7:
        level = "AAA"
    elif ratio >= 4.5:
        level = "AA"
    elif ratio >= 3:
        level = "AA-large"
    else:
        level = "fail"

    return ratio, level


def analyze_distribution(notable_colors: list) -> str:
    """Analyze how the color distribution compares to 60-30-10."""
    if not notable_colors:
        return "No significant colors detected"

    # Sort by coverage
    sorted_colors = sorted(notable_colors, key=lambda c: -c.coverage)

    dominant_coverage = sorted_colors[0].coverage * 100

    # Sum coverage by role
    accent_coverage = sum(c.coverage * 100 for c in notable_colors if c.role == 'accent')
    secondary_coverage = sum(c.coverage * 100 for c in notable_colors
                            if c.role in ('secondary', 'dark', 'light', 'dominant'))
    secondary_coverage -= dominant_coverage  # Don't double count dominant

    # Calculate top 3 coverage
    top3_coverage = sum(c.coverage * 100 for c in sorted_colors[:3])

    if dominant_coverage >= 15:
        if accent_coverage < 1:
            return f"Dominant-led: {dominant_coverage:.0f}% primary, {secondary_coverage:.0f}% supporting"
        else:
            return f"Structured: {dominant_coverage:.0f}% dominant, {secondary_coverage:.0f}% secondary, {accent_coverage:.1f}% accent"
    elif top3_coverage >= 40:
        return f"Tiered: Top 3 colors cover {top3_coverage:.0f}% of image"
    else:
        return f"Distributed: {len(notable_colors)} colors share coverage"


def compute_hue_structure(notable_colors: list[NotableColor]) -> HueStructure:
    """
    Compute hue structure from displayed notable colors.

    Derives chromatic family count, hue centers, and angular separations
    directly from the colors shown to the consumer.
    """
    # Group colors by family index
    families_by_idx: dict[int, list[NotableColor]] = {}
    for color in notable_colors:
        families_by_idx.setdefault(color.family_index, []).append(color)

    # Identify chromatic families (average chroma > threshold)
    chromatic_hues = []
    for colors in families_by_idx.values():
        avg_chroma = sum(c.chroma for c in colors) / len(colors)
        if avg_chroma >= NEUTRAL_CHROMA_THRESHOLD:
            # Compute coverage-weighted circular mean hue for this family
            hue_rads = [np.radians(compute_hue(c.lab)) for c in colors]
            sin_sum = sum(c.coverage * np.sin(h) for c, h in zip(colors, hue_rads))
            cos_sum = sum(c.coverage * np.cos(h) for c, h in zip(colors, hue_rads))
            avg_hue = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
            chromatic_hues.append(avg_hue)

    # Sort hues for consistent ordering
    chromatic_hues.sort()

    # Compute separations between adjacent hues (circular)
    separations = []
    if len(chromatic_hues) >= 2:
        for i in range(len(chromatic_hues)):
            next_i = (i + 1) % len(chromatic_hues)
            if next_i == 0:
                # Wrap-around: last to first
                sep = (chromatic_hues[0] + 360) - chromatic_hues[-1]
            else:
                sep = chromatic_hues[next_i] - chromatic_hues[i]
            separations.append(sep)

    return HueStructure(
        chromatic_count=len(chromatic_hues),
        hue_centers=chromatic_hues,
        separations=separations
    )


# Threshold for hue-based family clustering (degrees)
HUE_FAMILY_THRESHOLD = 45.0
# Chroma threshold below which colors are considered neutral (true grays)
NEUTRAL_CHROMA_THRESHOLD = 8.0


def assign_hue_families(colors: list['NotableColor']) -> None:
    """
    Assign family indices to colors based on hue clustering.

    Groups colors by hue proximity for presentation purposes.
    Neutrals (low chroma) form their own family.
    Families are ordered by total coverage (largest first).
    Mutates the family_index field of each color in place.
    """
    if not colors:
        return

    # Separate neutrals from chromatic colors
    neutrals = [c for c in colors if c.chroma < NEUTRAL_CHROMA_THRESHOLD]
    chromatic = [c for c in colors if c.chroma >= NEUTRAL_CHROMA_THRESHOLD]

    # Sort chromatic by coverage so high-coverage colors anchor families
    chromatic.sort(key=lambda c: -c.coverage)

    # Cluster chromatic colors by hue proximity
    hue_families: list[list['NotableColor']] = []

    for color in chromatic:
        hue = compute_hue(color.lab)

        # Find a family with similar hue
        matched = False
        for family in hue_families:
            # Compare to first member's hue (representative)
            family_hue = compute_hue(family[0].lab)
            if circular_hue_distance(hue, family_hue) <= HUE_FAMILY_THRESHOLD:
                family.append(color)
                matched = True
                break

        if not matched:
            hue_families.append([color])

    # Sort families by total coverage (largest first)
    hue_families.sort(key=lambda fam: -sum(c.coverage for c in fam))

    # Neutrals get their own family if present, placed after chromatic families
    all_families = hue_families + ([neutrals] if neutrals else [])

    # Assign family indices
    for family_idx, family in enumerate(all_families):
        for color in family:
            color.family_index = family_idx


def synthesize(data: PreparedData, features: FeatureData) -> SynthesisResult:
    """Stage 3: Synthesize features into actionable analysis."""

    # Family-based color selection
    # 1. Cluster colors into perceptually similar families
    family_clusters = cluster_into_families(features.metrics)
    families = analyze_families(features.metrics, family_clusters)

    # 2. Select colors ensuring each family gets representation
    selected_colors = select_colors_by_family(features.metrics, families)

    # Map family roles to output roles
    role_mapping = {
        'chromatic': 'secondary',
        'dark': 'dark',
        'light': 'light',
        'neutral_mid': 'secondary',
        'accent': 'accent',
        'fill': 'secondary'
    }

    # Build NotableColor objects
    notable_colors = []
    for i, (coarse_bin, family_role) in enumerate(selected_colors):
        metrics = features.metrics[coarse_bin]
        stab = features.stability.get(coarse_bin, StabilityInfo('texture', 0, 0))

        # First color by coverage is dominant
        if i == 0:
            role = 'dominant'
        else:
            role = role_mapping.get(family_role, 'secondary')

        # Count gradient membership
        gradient_count = sum(1 for g in features.gradients if coarse_bin in g.stops)

        notable_colors.append(NotableColor(
            coarse_bin=coarse_bin,
            lab=metrics.lab,
            hex=lab_to_hex(metrics.lab),
            rgb=lab_to_rgb_tuple(metrics.lab),
            name=generate_color_name(metrics.lab),
            role=role,
            coverage=metrics.coverage,
            chroma=metrics.chroma,
            significance=metrics.coverage * 100,  # Use coverage as significance proxy
            characteristics=generate_characteristics(metrics, stab, role, gradient_count),
            gradient_membership=[idx for idx, g in enumerate(features.gradients) if coarse_bin in g.stops]
        ))

    # Assign families by hue clustering (post-selection presentation grouping)
    # This groups colors by hue angle for display, separate from the LAB-based selection
    assign_hue_families(notable_colors)

    # Sort notable colors: by family, then by lightness (light to dark) within family
    notable_colors.sort(key=lambda c: (c.family_index, -c.lab[0]))

    # Find contrast pairs
    contrast_pairs = []
    for i, c1 in enumerate(notable_colors):
        for c2 in notable_colors[i+1:]:
            delta_l = abs(c1.lab[0] - c2.lab[0])
            if delta_l > 30:  # Meaningful contrast
                ratio, level = compute_wcag_contrast(c1.lab, c2.lab)
                contrast_pairs.append(ContrastPair(
                    color_a=c1.name,
                    color_b=c2.name,
                    delta_l=delta_l,
                    contrast_ratio=ratio,
                    wcag_level=level
                ))

    contrast_pairs.sort(key=lambda p: -p.contrast_ratio)
    contrast_pairs = contrast_pairs[:5]  # Top 5

    # Find harmonic pairs (similar hue) - limit to pairs with different names
    harmonic_pairs = []
    chromatic_notable = [c for c in notable_colors if c.chroma >= 15]
    seen_pairs = set()

    for i, c1 in enumerate(chromatic_notable):
        for c2 in chromatic_notable[i+1:]:
            # Skip if same name (different shades of same color)
            if c1.name == c2.name:
                continue

            hue1 = compute_hue(c1.lab)
            hue2 = compute_hue(c2.lab)
            hue_diff = circular_hue_distance(hue1, hue2)

            if hue_diff < HUE_CLUSTER_RANGE:
                # Deduplicate by name pair
                pair_key = tuple(sorted([c1.name, c2.name]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    harmonic_pairs.append(HarmonicPair(
                        color_a=c1.name,
                        color_b=c2.name,
                        hue_difference=hue_diff
                    ))

    harmonic_pairs = harmonic_pairs[:5]  # Limit to top 5

    # Compute hue structure from displayed notable colors
    hue_structure = compute_hue_structure(notable_colors)

    # Lightness and chroma ranges
    all_L = [m.lab[0] for m in features.metrics.values()]
    all_chroma = [m.chroma for m in features.metrics.values()]

    # Count unique families represented in notable colors
    family_count = len(set(c.family_index for c in notable_colors))

    return SynthesisResult(
        hue_structure=hue_structure,
        notable_colors=notable_colors,
        gradients=features.gradients,
        contrast_pairs=contrast_pairs,
        harmonic_pairs=harmonic_pairs,
        distribution_analysis=analyze_distribution(notable_colors),
        lightness_range=(min(all_L), max(all_L)),
        chroma_range=(min(all_chroma), max(all_chroma)),
        family_count=family_count
    )


# =============================================================================
# Stage 4: Render
# =============================================================================

def render(synthesis: SynthesisResult, features: FeatureData) -> str:
    """Stage 4: Render synthesis result as prose."""
    lines = []

    # Header
    lines.append("OVERVIEW:")
    lines.append("Palette analysis based on hue distribution, contrast range, and color")
    lines.append("relationships. Lightness (L) and chroma (saturation) ranges show tonal spread.")
    lines.append("")

    # Hue structure
    hs = synthesis.hue_structure
    if hs.chromatic_count == 0:
        lines.append("Hue structure: achromatic (no chromatic families)")
    elif hs.chromatic_count == 1:
        lines.append(f"Hue structure: 1 chromatic family at {hs.hue_centers[0]:.0f}°")
    else:
        hues_str = ", ".join(f"{h:.0f}°" for h in hs.hue_centers)
        seps_str = ", ".join(f"{s:.0f}°" for s in hs.separations)
        lines.append(f"Hue structure: {hs.chromatic_count} chromatic families at {hues_str}")
        lines.append(f"  Separations: {seps_str}")
    lines.append(f"Lightness range: {synthesis.lightness_range[0]:.0f}-{synthesis.lightness_range[1]:.0f} | "
                 f"Chroma range: {synthesis.chroma_range[0]:.0f}-{synthesis.chroma_range[1]:.0f}")
    lines.append(f"Notable colors: {len(synthesis.notable_colors)}")
    lines.append(f"Distribution: {synthesis.distribution_analysis}")
    lines.append("")

    # Colors section - grouped by family, sorted light→dark within each
    lines.append("COLORS:")
    lines.append(f"{len(synthesis.notable_colors)} colors across {synthesis.family_count} families, sorted light→dark within each.")
    lines.append("")

    # Group colors by family
    current_family = -1
    for color in synthesis.notable_colors:
        if color.family_index != current_family:
            if current_family != -1:
                lines.append("")  # Blank line between families
            current_family = color.family_index
            # Count colors in this family
            family_size = sum(1 for c in synthesis.notable_colors if c.family_index == current_family)
            lines.append(f"Family {current_family + 1} ({family_size} {'color' if family_size == 1 else 'colors'}):")

        coverage_pct = color.coverage * 100
        coverage_str = f"{coverage_pct:.1f}%" if coverage_pct >= 0.1 else "<0.1%"
        lines.append(f"  [{color.role.capitalize()}] {color.name} {color.hex} | L={color.lab[0]:.0f} | {coverage_str}")
    lines.append("")

    # Gradients section
    if synthesis.gradients:
        lines.append("GRADIENTS:")
        lines.append("Smooth color transitions detected in the image. Gradients indicate areas")
        lines.append("where colors blend spatially — useful for backgrounds or lighting effects.")
        lines.append("Direction shows transition axis; span measures lightness range across stops.")
        lines.append("")

        for i, grad in enumerate(synthesis.gradients):
            # Build stop names
            stop_names = []
            for stop in grad.stops:
                metrics = features.metrics.get(stop)
                if metrics:
                    stop_names.append(generate_color_name(metrics.lab))
                else:
                    stop_names.append("Unknown")

            lines.append(f"{' → '.join(stop_names)}")
            lines.append(f"  Stops:")

            for j, stop in enumerate(grad.stops):
                metrics = features.metrics.get(stop)
                if metrics:
                    hex_val = lab_to_hex(metrics.lab)
                    rgb_val = lab_to_rgb_tuple(metrics.lab)
                    lab_val = metrics.lab
                    name = generate_color_name(lab_val)
                    lines.append(f"    {j+1}. {name} {hex_val} / RGB{rgb_val} / LAB({lab_val[0]:.0f}, {lab_val[1]:.0f}, {lab_val[2]:.0f})")

            lines.append(f"  Direction: {grad.direction} | Coverage: {grad.coverage*100:.1f}%")

            L_range = grad.lab_range['L'][1] - grad.lab_range['L'][0]
            lines.append(f"  Lightness span: {L_range:.0f} (L {grad.lab_range['L'][0]:.0f} → {grad.lab_range['L'][1]:.0f})")
            lines.append("")

    # Relationships section
    lines.append("RELATIONSHIPS:")
    lines.append("How colors interact. Contrast pairs have high visual separation — rated by")
    lines.append("WCAG accessibility (AAA ≥7:1, AA ≥4.5:1, AA-large ≥3:1). Harmonic pairs share")
    lines.append("similar hue and naturally complement each other.")
    lines.append("")

    if synthesis.contrast_pairs:
        lines.append("Contrast pairs (high separation — good for text/emphasis):")
        for pair in synthesis.contrast_pairs:
            lines.append(f"  - {pair.color_a} / {pair.color_b}: "
                        f"Ratio {pair.contrast_ratio:.1f}:1 (WCAG {pair.wcag_level}) | ΔL={pair.delta_l:.0f}")
        lines.append("")

    if synthesis.harmonic_pairs:
        lines.append("Harmonic pairs (similar hue — good for cohesive backgrounds):")
        for pair in synthesis.harmonic_pairs:
            lines.append(f"  - {pair.color_a} and {pair.color_b}: "
                        f"Similar hue ({pair.hue_difference:.0f}° apart)")

    return "\n".join(lines)


def text_color_for_background(L: float) -> str:
    """Return black or white text color based on background lightness."""
    return "#000" if L > 50 else "#fff"


def render_html(synthesis: SynthesisResult, features: FeatureData, image_path: str) -> str:
    """Stage 4b: Render synthesis result as HTML."""
    safe_path = html_escape(image_path)

    # Build name lookups once for contrast/harmonic pairs
    name_to_hex = {c.name: c.hex for c in synthesis.notable_colors}
    name_to_L = {c.name: c.lab[0] for c in synthesis.notable_colors}

    # CSS styles
    css = """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: system-ui, -apple-system, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.5;
            padding: 2rem;
            max-width: 900px;
            margin: 0 auto;
        }
        h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
        h2 { font-size: 1.2rem; margin: 2rem 0 1rem; border-bottom: 1px solid #ddd; padding-bottom: 0.5rem; }
        .meta { color: #666; font-size: 0.9rem; margin-bottom: 1rem; }
        .palette-strip {
            display: flex;
            height: 80px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 1.5rem 0;
        }
        .palette-strip .swatch {
            display: flex;
            align-items: flex-end;
            justify-content: center;
            padding: 0.5rem;
            font-size: 0.7rem;
            font-weight: 500;
        }
        .color-card {
            background: #fff;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            display: grid;
            grid-template-columns: 60px 1fr;
            gap: 1rem;
        }
        .color-card .swatch {
            width: 60px;
            height: 60px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.65rem;
            font-weight: 600;
        }
        .color-card .info { font-size: 0.85rem; }
        .color-card .role { font-weight: 600; text-transform: capitalize; }
        .color-card .name { color: #666; }
        .color-card .values { font-family: monospace; color: #555; font-size: 0.8rem; }
        .color-card .chars { font-style: italic; color: #777; margin-top: 0.25rem; }
        .family-group {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            background: #fafafa;
        }
        .family-group .color-card { margin-bottom: 0.75rem; }
        .family-group .color-card:last-child { margin-bottom: 0; }
        .family-header {
            font-size: 0.9rem;
            font-weight: 600;
            color: #555;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e8e8e8;
        }
        .gradient-block {
            background: #fff;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .gradient-bar {
            height: 40px;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .gradient-stops {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .gradient-stop {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
        }
        .gradient-stop .swatch {
            width: 24px;
            height: 24px;
            border-radius: 4px;
        }
        .gradient-meta { font-size: 0.85rem; color: #666; margin-top: 0.75rem; }
        .contrast-pair {
            background: #fff;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .contrast-demo {
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }
        .contrast-demo .sample { font-weight: 600; }
        .contrast-info { font-size: 0.85rem; color: #666; }
        .contrast-badge {
            display: inline-block;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }
        .badge-aaa { background: #22c55e; color: #fff; }
        .badge-aa { background: #3b82f6; color: #fff; }
        .badge-aa-large { background: #f59e0b; color: #fff; }
        .badge-fail { background: #ef4444; color: #fff; }
        .harmonic-pair {
            background: #fff;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .harmonic-swatches {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }
        .harmonic-swatches .swatch {
            flex: 1;
            height: 48px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 500;
        }
        .harmonic-info {
            font-size: 0.85rem;
            color: #666;
        }
        .harmonic-info .names {
            color: #333;
            font-weight: 500;
        }
        .section-intro { font-size: 0.9rem; color: #666; margin-bottom: 1rem; line-height: 1.6; }
    """

    # Build HTML
    lines = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '  <meta charset="UTF-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'  <title>Palette: {safe_path}</title>',
        f'  <style>{css}</style>',
        '</head>',
        '<body>',
    ]

    # Header
    lines.append('<h2>Overview</h2>')
    lines.append('<p class="section-intro">Palette analysis based on hue distribution, contrast range, and color '
                 'relationships. Lightness (L) and chroma (saturation) ranges show tonal spread.</p>')

    # Hue structure
    hs = synthesis.hue_structure
    if hs.chromatic_count == 0:
        hue_title = "Achromatic"
        hue_desc = "No chromatic families"
    elif hs.chromatic_count == 1:
        hue_title = f"1 Chromatic Family"
        hue_desc = f"Hue center at {hs.hue_centers[0]:.0f}°"
    else:
        hues_str = ", ".join(f"{h:.0f}°" for h in hs.hue_centers)
        seps_str = ", ".join(f"{s:.0f}°" for s in hs.separations)
        hue_title = f"{hs.chromatic_count} Chromatic Families"
        hue_desc = f"Hue centers: {hues_str} | Separations: {seps_str}"

    lines.append(f'<h1>{hue_title}</h1>')
    lines.append(f'<p class="meta">{hue_desc}</p>')
    lines.append(f'<p class="meta">Source: {safe_path}</p>')
    lines.append(f'<p class="meta">Lightness: {synthesis.lightness_range[0]:.0f}–{synthesis.lightness_range[1]:.0f} | '
                 f'Chroma: {synthesis.chroma_range[0]:.0f}–{synthesis.chroma_range[1]:.0f} | '
                 f'{len(synthesis.notable_colors)} colors</p>')
    lines.append(f'<p class="meta">Distribution: {synthesis.distribution_analysis}</p>')

    # Palette strip
    lines.append('<div class="palette-strip">')
    total_coverage = sum(c.coverage for c in synthesis.notable_colors) or 1
    for color in synthesis.notable_colors:
        width_pct = max(5, (color.coverage / total_coverage) * 100)  # min 5% for visibility
        text_color = text_color_for_background(color.lab[0])
        lines.append(f'  <div class="swatch" style="background:{color.hex}; color:{text_color}; flex:{width_pct:.1f}">{color.hex}</div>')
    lines.append('</div>')

    # Color details - grouped by family, sorted light→dark within each
    lines.append('<h2>Colors</h2>')
    lines.append(f'<p class="section-intro">{len(synthesis.notable_colors)} colors across '
                 f'{synthesis.family_count} families, sorted light→dark within each. '
                 '<strong>Roles:</strong> Dominant = highest coverage, '
                 'Secondary = supporting, Accent = high contrast/saturation, '
                 'Dark/Light = value extremes.</p>')

    # Group colors by family
    current_family = -1
    for color in synthesis.notable_colors:
        if color.family_index != current_family:
            if current_family != -1:
                lines.append('</div>')  # Close previous family group
            current_family = color.family_index
            family_size = sum(1 for c in synthesis.notable_colors if c.family_index == current_family)
            lines.append('<div class="family-group">')
            lines.append(f'  <div class="family-header">Family {current_family + 1} ({family_size} {"color" if family_size == 1 else "colors"})</div>')

        text_color = text_color_for_background(color.lab[0])
        coverage_str = f"{color.coverage*100:.1f}%" if color.coverage >= 0.001 else "<0.1%"
        lines.append('  <div class="color-card">')
        lines.append(f'    <div class="swatch" style="background:{color.hex}; color:{text_color}">{color.hex}</div>')
        lines.append('    <div class="info">')
        lines.append(f'      <span class="role">{color.role}</span> <span class="name">{color.name}</span>')
        lines.append(f'      <div class="values">RGB{color.rgb} · LAB({color.lab[0]:.0f}, {color.lab[1]:.0f}, {color.lab[2]:.0f})</div>')
        lines.append(f'      <div class="values">Coverage: {coverage_str} · Chroma: {color.chroma:.0f}</div>')
        if color.characteristics:
            lines.append(f'      <div class="chars">{". ".join(color.characteristics)}.</div>')
        lines.append('    </div>')
        lines.append('  </div>')

    if current_family != -1:
        lines.append('</div>')  # Close last family group

    # Gradients
    if synthesis.gradients:
        lines.append('<h2>Gradients</h2>')
        lines.append('<p class="section-intro">Smooth color transitions detected in the image. '
                     'Gradients indicate areas where colors blend spatially — useful for background design or identifying lighting effects. '
                     'Direction shows the transition axis; span measures the lightness range across stops.</p>')
        fine_size = FINE_SCALE * JND

        for grad in synthesis.gradients:
            # Subsample fine_members for cleaner CSS gradient (max ~20 stops)
            max_css_stops = 20
            fine_members = grad.fine_members
            if len(fine_members) > max_css_stops:
                # Evenly sample across the chain
                indices = [int(i * (len(fine_members) - 1) / (max_css_stops - 1))
                          for i in range(max_css_stops)]
                fine_members = [fine_members[i] for i in indices]

            # Build CSS gradient from subsampled fine_members
            fine_stops_css = []
            for i, fine_bin in enumerate(fine_members):
                lab = np.array(fine_bin) * fine_size
                hex_val = lab_to_hex(lab)
                pct = (i / max(1, len(fine_members) - 1)) * 100
                fine_stops_css.append(f"{hex_val} {pct:.0f}%")

            # Build labeled stops from coarse bins (representative colors)
            stop_info = []
            for stop in grad.stops:
                metrics = features.metrics.get(stop)
                if metrics:
                    hex_val = lab_to_hex(metrics.lab)
                    name = generate_color_name(metrics.lab)
                    stop_info.append((hex_val, name, metrics.lab))

            # Skip if no valid stops found
            if not fine_stops_css:
                continue

            # Always render horizontally for easier visual comparison
            # (actual spatial direction shown in metadata below)
            gradient_css = f"linear-gradient(to right, {', '.join(fine_stops_css)})"

            lines.append('<div class="gradient-block">')
            lines.append(f'  <div class="gradient-bar" style="background:{gradient_css}"></div>')
            lines.append('  <div class="gradient-stops">')
            for hex_val, name, _ in stop_info:
                lines.append(f'    <div class="gradient-stop"><div class="swatch" style="background:{hex_val}"></div>{name}</div>')
            lines.append('  </div>')
            L_range = grad.lab_range['L'][1] - grad.lab_range['L'][0]
            lines.append(f'  <div class="gradient-meta">Direction: {grad.direction} · Coverage: {grad.coverage*100:.1f}% · '
                        f'Lightness span: {L_range:.0f}</div>')
            lines.append('</div>')

    # Relationships
    lines.append('<h2>Relationships</h2>')
    lines.append('<p class="section-intro">How colors interact. '
                 '<strong>Contrast pairs</strong> have high visual separation — rated by WCAG accessibility standards '
                 '(AAA ≥7:1, AA ≥4.5:1, AA-large ≥3:1). '
                 '<strong>Harmonic pairs</strong> share similar hue and naturally complement each other.</p>')

    # Contrast pairs
    if synthesis.contrast_pairs:
        lines.append('<h3 style="font-size:1rem; margin:1rem 0 0.5rem;">Contrast Pairs</h3>')
        for pair in synthesis.contrast_pairs:
            bg_hex = name_to_hex.get(pair.color_a, '#888')
            fg_hex = name_to_hex.get(pair.color_b, '#fff')

            badge_class = {
                'AAA': 'badge-aaa',
                'AA': 'badge-aa',
                'AA-large': 'badge-aa-large',
            }.get(pair.wcag_level, 'badge-fail')

            lines.append('<div class="contrast-pair">')
            lines.append(f'  <div class="contrast-demo" style="background:{bg_hex}; color:{fg_hex}">')
            lines.append(f'    <span class="sample">Aa</span> Sample text for readability')
            lines.append('  </div>')
            lines.append(f'  <div class="contrast-info">{pair.color_a} / {pair.color_b}: '
                        f'{pair.contrast_ratio:.1f}:1 <span class="contrast-badge {badge_class}">{pair.wcag_level}</span></div>')
            lines.append('</div>')

    # Harmonic pairs
    if synthesis.harmonic_pairs:
        lines.append('<h3 style="font-size:1rem; margin:1rem 0 0.5rem;">Harmonic Pairs</h3>')
        for pair in synthesis.harmonic_pairs:
            hex_a = name_to_hex.get(pair.color_a, '#888')
            hex_b = name_to_hex.get(pair.color_b, '#888')
            L_a = name_to_L.get(pair.color_a, 50)
            L_b = name_to_L.get(pair.color_b, 50)
            text_a = text_color_for_background(L_a)
            text_b = text_color_for_background(L_b)
            lines.append('<div class="harmonic-pair">')
            lines.append('  <div class="harmonic-swatches">')
            lines.append(f'    <div class="swatch" style="background:{hex_a}; color:{text_a}">{hex_a}</div>')
            lines.append(f'    <div class="swatch" style="background:{hex_b}; color:{text_b}">{hex_b}</div>')
            lines.append('  </div>')
            lines.append(f'  <div class="harmonic-info"><span class="names">{pair.color_a}</span> and '
                        f'<span class="names">{pair.color_b}</span>: {pair.hue_difference:.0f}° apart</div>')
            lines.append('</div>')

    lines.append('</body>')
    lines.append('</html>')

    return '\n'.join(lines)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(image_path: str, downscale: bool = True) -> tuple[SynthesisResult, FeatureData]:
    """Run analysis pipeline stages 1-3.

    Args:
        image_path: Path to the image file
        downscale: If True, resize images so longest edge is 256px (default: True)

    Returns:
        Tuple of (synthesis_result, feature_data) for rendering.
    """
    # Stage 1: Data Preparation
    data = prepare_data(image_path, downscale=downscale)

    # Stage 2: Feature Extraction
    features = extract_features(data)

    # Stage 3: Synthesis
    synthesis = synthesize(data, features)

    return synthesis, features


def analyze_image(image_path: str, downscale: bool = True) -> tuple[str, str]:
    """Run the full analysis pipeline on an image.

    Args:
        image_path: Path to the image file
        downscale: If True, resize images so longest edge is 256px (default: True)

    Returns:
        Tuple of (prose_output, html_output)
    """
    synthesis, features = run_pipeline(image_path, downscale=downscale)
    prose = render(synthesis, features)
    html = render_html(synthesis, features, image_path)
    return prose, html


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point for palette extraction."""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Analyze an image and extract its color palette.'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to the image file'
    )
    parser.add_argument(
        '--output', '-o',
        nargs='?',
        const=True,
        default=None,
        help='Write HTML report. Optionally specify path, otherwise auto-names from input.'
    )
    parser.add_argument(
        '--no-downscale',
        action='store_true',
        help='Process at full resolution instead of downscaling to 256px'
    )

    args = parser.parse_args()
    image_path = Path(args.input)

    # Run analysis
    try:
        prose, html = analyze_image(str(image_path), downscale=not args.no_downscale)
    except KeyboardInterrupt:
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing image: {e}", file=sys.stderr)
        sys.exit(1)

    # Always print prose to terminal
    print(prose)

    # Write HTML if requested
    if args.output:
        if args.output is True:
            output_path = image_path.with_name(f"{image_path.stem}-palette.html")
        else:
            output_path = Path(args.output)

        try:
            output_path.write_text(html)
            print(f"\nWrote: {output_path}")
        except OSError as e:
            print(f"Error writing output: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
