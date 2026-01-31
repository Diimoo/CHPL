#!/usr/bin/env python3
"""
Extended Dictionary for Language Bootstrapping

Creates a large dictionary (~2000 entries) for vocabulary expansion.
Each definition only uses simple, grounded words.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DictionaryEntry:
    word: str
    definition: str
    category: str = ""


def create_extended_color_dictionary() -> List[DictionaryEntry]:
    """Extended color vocabulary."""
    return [
        # Reds
        DictionaryEntry("crimson", "a deep red color", "color"),
        DictionaryEntry("scarlet", "a bright red color", "color"),
        DictionaryEntry("ruby", "a deep red like the stone", "color"),
        DictionaryEntry("maroon", "a dark red color", "color"),
        DictionaryEntry("burgundy", "a dark red purple color", "color"),
        DictionaryEntry("cherry", "a bright red color", "color"),
        DictionaryEntry("rose", "a light red pink color", "color"),
        DictionaryEntry("salmon", "a pink orange color", "color"),
        DictionaryEntry("coral", "a pink orange red color", "color"),
        DictionaryEntry("rust", "a red orange brown color", "color"),
        
        # Blues
        DictionaryEntry("navy", "a dark blue color", "color"),
        DictionaryEntry("azure", "a bright blue color", "color"),
        DictionaryEntry("cobalt", "a deep blue color", "color"),
        DictionaryEntry("sapphire", "a deep blue like the stone", "color"),
        DictionaryEntry("sky", "a light blue color", "color"),
        DictionaryEntry("ocean", "a blue green color", "color"),
        DictionaryEntry("midnight", "a very dark blue color", "color"),
        DictionaryEntry("royal", "a deep bright blue color", "color"),
        DictionaryEntry("powder", "a very light blue color", "color"),
        DictionaryEntry("steel", "a gray blue color", "color"),
        
        # Greens
        DictionaryEntry("emerald", "a bright green color", "color"),
        DictionaryEntry("olive", "a yellow green color", "color"),
        DictionaryEntry("lime", "a bright yellow green color", "color"),
        DictionaryEntry("forest", "a dark green color", "color"),
        DictionaryEntry("mint", "a light green color", "color"),
        DictionaryEntry("jade", "a green blue color", "color"),
        DictionaryEntry("sage", "a gray green color", "color"),
        DictionaryEntry("moss", "a dark yellow green color", "color"),
        DictionaryEntry("grass", "a bright green color", "color"),
        DictionaryEntry("pine", "a dark green color", "color"),
        
        # Yellows and Oranges
        DictionaryEntry("gold", "a bright yellow color", "color"),
        DictionaryEntry("amber", "a warm yellow orange color", "color"),
        DictionaryEntry("honey", "a golden yellow color", "color"),
        DictionaryEntry("lemon", "a bright yellow color", "color"),
        DictionaryEntry("mustard", "a dark yellow color", "color"),
        DictionaryEntry("tangerine", "a bright orange color", "color"),
        DictionaryEntry("peach", "a light orange pink color", "color"),
        DictionaryEntry("apricot", "a yellow orange color", "color"),
        DictionaryEntry("copper", "a red orange brown color", "color"),
        DictionaryEntry("bronze", "a brown orange color", "color"),
        
        # Purples
        DictionaryEntry("violet", "a blue purple color", "color"),
        DictionaryEntry("lavender", "a light purple color", "color"),
        DictionaryEntry("indigo", "a deep blue purple color", "color"),
        DictionaryEntry("magenta", "a red purple color", "color"),
        DictionaryEntry("plum", "a dark purple color", "color"),
        DictionaryEntry("grape", "a purple color", "color"),
        DictionaryEntry("lilac", "a light purple color", "color"),
        DictionaryEntry("orchid", "a light purple pink color", "color"),
        DictionaryEntry("mauve", "a gray purple color", "color"),
        DictionaryEntry("amethyst", "a purple color like the stone", "color"),
        
        # Neutrals
        DictionaryEntry("white", "the lightest color", "color"),
        DictionaryEntry("black", "the darkest color", "color"),
        DictionaryEntry("gray", "a color between white and black", "color"),
        DictionaryEntry("silver", "a light gray color", "color"),
        DictionaryEntry("charcoal", "a dark gray color", "color"),
        DictionaryEntry("ivory", "a white yellow color", "color"),
        DictionaryEntry("cream", "a white yellow color", "color"),
        DictionaryEntry("beige", "a light brown color", "color"),
        DictionaryEntry("tan", "a light brown color", "color"),
        DictionaryEntry("brown", "a dark orange color", "color"),
    ]


def create_extended_shape_dictionary() -> List[DictionaryEntry]:
    """Extended shape vocabulary."""
    return [
        # 2D shapes
        DictionaryEntry("oval", "a shape like a circle stretched", "shape"),
        DictionaryEntry("ellipse", "a shape like a circle but longer", "shape"),
        DictionaryEntry("rectangle", "a shape like a square but longer", "shape"),
        DictionaryEntry("diamond", "a square shape turned", "shape"),
        DictionaryEntry("rhombus", "a shape like a diamond", "shape"),
        DictionaryEntry("pentagon", "a shape with five sides", "shape"),
        DictionaryEntry("hexagon", "a shape with six sides", "shape"),
        DictionaryEntry("heptagon", "a shape with seven sides", "shape"),
        DictionaryEntry("octagon", "a shape with eight sides", "shape"),
        DictionaryEntry("polygon", "a shape with many sides", "shape"),
        DictionaryEntry("crescent", "a shape like a curved moon", "shape"),
        DictionaryEntry("heart", "a shape with two curves above", "shape"),
        DictionaryEntry("arrow", "a triangle with a line", "shape"),
        DictionaryEntry("cross", "two lines crossing", "shape"),
        DictionaryEntry("ring", "a circle with a hole inside", "shape"),
        
        # 3D shapes
        DictionaryEntry("sphere", "a circle in three dimensions", "shape"),
        DictionaryEntry("cube", "a square in three dimensions", "shape"),
        DictionaryEntry("pyramid", "a triangle in three dimensions", "shape"),
        DictionaryEntry("cylinder", "a circle stretched upward", "shape"),
        DictionaryEntry("cone", "a circle with a point above", "shape"),
        DictionaryEntry("prism", "a shape with flat sides", "shape"),
        DictionaryEntry("dome", "half of a sphere", "shape"),
        DictionaryEntry("disk", "a flat circle", "shape"),
        DictionaryEntry("tube", "a long cylinder with a hole", "shape"),
        DictionaryEntry("spiral", "a line going in a circle outward", "shape"),
        
        # Shape properties
        DictionaryEntry("curved", "not straight like a circle", "shape"),
        DictionaryEntry("straight", "not curved like a line", "shape"),
        DictionaryEntry("pointed", "having a sharp end", "shape"),
        DictionaryEntry("rounded", "having curved edges", "shape"),
        DictionaryEntry("angular", "having many corners", "shape"),
        DictionaryEntry("symmetric", "the same on both sides", "shape"),
        DictionaryEntry("irregular", "not having a regular shape", "shape"),
        DictionaryEntry("hollow", "empty inside", "shape"),
        DictionaryEntry("solid", "not hollow", "shape"),
        DictionaryEntry("flat", "having no height", "shape"),
    ]


def create_extended_spatial_dictionary() -> List[DictionaryEntry]:
    """Extended spatial relations vocabulary."""
    return [
        # Vertical
        DictionaryEntry("atop", "on top of something", "spatial"),
        DictionaryEntry("over", "above something", "spatial"),
        DictionaryEntry("beneath", "below something", "spatial"),
        DictionaryEntry("under", "below something", "spatial"),
        DictionaryEntry("underneath", "directly below something", "spatial"),
        DictionaryEntry("overhead", "above and over something", "spatial"),
        
        # Horizontal
        DictionaryEntry("beside", "next to something", "spatial"),
        DictionaryEntry("alongside", "next to and along something", "spatial"),
        DictionaryEntry("adjacent", "next to something", "spatial"),
        DictionaryEntry("opposite", "on the other side", "spatial"),
        DictionaryEntry("facing", "looking toward something", "spatial"),
        
        # Distance
        DictionaryEntry("near", "close to something", "spatial"),
        DictionaryEntry("close", "near to something", "spatial"),
        DictionaryEntry("far", "not near something", "spatial"),
        DictionaryEntry("distant", "far from something", "spatial"),
        DictionaryEntry("nearby", "near and close", "spatial"),
        DictionaryEntry("remote", "very far away", "spatial"),
        
        # Containment
        DictionaryEntry("inside", "within something", "spatial"),
        DictionaryEntry("outside", "not inside something", "spatial"),
        DictionaryEntry("within", "inside of something", "spatial"),
        DictionaryEntry("without", "outside of something", "spatial"),
        DictionaryEntry("among", "in the middle of many things", "spatial"),
        DictionaryEntry("between", "in the middle of two things", "spatial"),
        DictionaryEntry("amid", "in the middle of many things", "spatial"),
        DictionaryEntry("surrounded", "with things all around", "spatial"),
        
        # Direction
        DictionaryEntry("toward", "in the direction of", "spatial"),
        DictionaryEntry("away", "in the opposite direction", "spatial"),
        DictionaryEntry("forward", "toward the front", "spatial"),
        DictionaryEntry("backward", "toward the back", "spatial"),
        DictionaryEntry("upward", "toward the top", "spatial"),
        DictionaryEntry("downward", "toward the bottom", "spatial"),
        DictionaryEntry("inward", "toward the inside", "spatial"),
        DictionaryEntry("outward", "toward the outside", "spatial"),
        DictionaryEntry("sideways", "toward the side", "spatial"),
        DictionaryEntry("diagonal", "not straight up or across", "spatial"),
        
        # Movement paths
        DictionaryEntry("through", "from one side to the other", "spatial"),
        DictionaryEntry("across", "from one side to the other", "spatial"),
        DictionaryEntry("around", "in a circle path", "spatial"),
        DictionaryEntry("along", "following a path", "spatial"),
        DictionaryEntry("past", "going by something", "spatial"),
        DictionaryEntry("beyond", "on the other side of", "spatial"),
    ]


def create_extended_action_dictionary() -> List[DictionaryEntry]:
    """Extended action vocabulary."""
    return [
        # Movement
        DictionaryEntry("move", "to change position", "action"),
        DictionaryEntry("push", "to move something away", "action"),
        DictionaryEntry("pull", "to move something toward", "action"),
        DictionaryEntry("lift", "to move something above", "action"),
        DictionaryEntry("drop", "to let something fall", "action"),
        DictionaryEntry("throw", "to push something fast", "action"),
        DictionaryEntry("catch", "to stop something moving", "action"),
        DictionaryEntry("hold", "to keep something still", "action"),
        DictionaryEntry("release", "to let something go", "action"),
        DictionaryEntry("grab", "to take and hold something", "action"),
        DictionaryEntry("place", "to put something down", "action"),
        DictionaryEntry("set", "to put something in a position", "action"),
        
        # Rotation
        DictionaryEntry("rotate", "to turn in a circle", "action"),
        DictionaryEntry("spin", "to rotate fast", "action"),
        DictionaryEntry("turn", "to rotate a small amount", "action"),
        DictionaryEntry("twist", "to rotate with force", "action"),
        DictionaryEntry("flip", "to turn upside down", "action"),
        DictionaryEntry("roll", "to rotate while moving", "action"),
        
        # Speed
        DictionaryEntry("accelerate", "to move faster", "action"),
        DictionaryEntry("decelerate", "to move slower", "action"),
        DictionaryEntry("speed", "to move fast", "action"),
        DictionaryEntry("slow", "to move less fast", "action"),
        DictionaryEntry("rush", "to move very fast", "action"),
        DictionaryEntry("crawl", "to move very slow", "action"),
        
        # State changes
        DictionaryEntry("stop", "to not move", "action"),
        DictionaryEntry("start", "to begin to move", "action"),
        DictionaryEntry("pause", "to stop for a short time", "action"),
        DictionaryEntry("continue", "to keep moving", "action"),
        DictionaryEntry("resume", "to start again after stopping", "action"),
        
        # Collision
        DictionaryEntry("hit", "to touch with force", "action"),
        DictionaryEntry("strike", "to hit something hard", "action"),
        DictionaryEntry("bump", "to hit something lightly", "action"),
        DictionaryEntry("crash", "to hit something with great force", "action"),
        DictionaryEntry("bounce", "to move away after hitting", "action"),
        DictionaryEntry("collide", "to hit another moving thing", "action"),
        
        # Transformation
        DictionaryEntry("grow", "to become larger", "action"),
        DictionaryEntry("shrink", "to become smaller", "action"),
        DictionaryEntry("expand", "to grow outward", "action"),
        DictionaryEntry("contract", "to shrink inward", "action"),
        DictionaryEntry("stretch", "to become longer", "action"),
        DictionaryEntry("compress", "to become shorter", "action"),
        DictionaryEntry("bend", "to become curved", "action"),
        DictionaryEntry("straighten", "to become straight", "action"),
        
        # Creation and destruction
        DictionaryEntry("create", "to make something new", "action"),
        DictionaryEntry("destroy", "to break something completely", "action"),
        DictionaryEntry("build", "to create by putting together", "action"),
        DictionaryEntry("break", "to separate into parts", "action"),
        DictionaryEntry("combine", "to put things together", "action"),
        DictionaryEntry("separate", "to move things apart", "action"),
        DictionaryEntry("connect", "to join things together", "action"),
        DictionaryEntry("disconnect", "to separate connected things", "action"),
    ]


def create_extended_property_dictionary() -> List[DictionaryEntry]:
    """Extended property vocabulary."""
    return [
        # Size
        DictionaryEntry("big", "large in size", "property"),
        DictionaryEntry("small", "not large in size", "property"),
        DictionaryEntry("large", "big in size", "property"),
        DictionaryEntry("tiny", "very small in size", "property"),
        DictionaryEntry("huge", "very large in size", "property"),
        DictionaryEntry("giant", "extremely large", "property"),
        DictionaryEntry("miniature", "extremely small", "property"),
        DictionaryEntry("medium", "between small and large", "property"),
        DictionaryEntry("massive", "very very large", "property"),
        DictionaryEntry("microscopic", "too small to see", "property"),
        
        # Speed
        DictionaryEntry("fast", "moving quickly", "property"),
        DictionaryEntry("quick", "moving fast", "property"),
        DictionaryEntry("rapid", "very fast", "property"),
        DictionaryEntry("swift", "moving very fast", "property"),
        DictionaryEntry("sluggish", "moving very slow", "property"),
        DictionaryEntry("steady", "moving at the same speed", "property"),
        
        # Weight
        DictionaryEntry("heavy", "having much weight", "property"),
        DictionaryEntry("light", "having little weight", "property"),
        DictionaryEntry("dense", "heavy for its size", "property"),
        DictionaryEntry("weightless", "having no weight", "property"),
        
        # Temperature
        DictionaryEntry("hot", "having much heat", "property"),
        DictionaryEntry("cold", "having little heat", "property"),
        DictionaryEntry("warm", "having some heat", "property"),
        DictionaryEntry("cool", "having less heat than warm", "property"),
        DictionaryEntry("freezing", "very very cold", "property"),
        DictionaryEntry("boiling", "very very hot", "property"),
        
        # Texture
        DictionaryEntry("smooth", "having no bumps", "property"),
        DictionaryEntry("rough", "having many bumps", "property"),
        DictionaryEntry("soft", "easy to press", "property"),
        DictionaryEntry("hard", "difficult to press", "property"),
        DictionaryEntry("sticky", "things attach to it", "property"),
        DictionaryEntry("slippery", "things slide on it", "property"),
        
        # Light
        DictionaryEntry("bright", "having much light", "property"),
        DictionaryEntry("dark", "having little light", "property"),
        DictionaryEntry("shiny", "reflecting light", "property"),
        DictionaryEntry("dull", "not reflecting light", "property"),
        DictionaryEntry("transparent", "light passes through", "property"),
        DictionaryEntry("opaque", "light does not pass through", "property"),
        
        # Dimensions
        DictionaryEntry("tall", "having much height", "property"),
        DictionaryEntry("short", "having little height", "property"),
        DictionaryEntry("wide", "having much width", "property"),
        DictionaryEntry("narrow", "having little width", "property"),
        DictionaryEntry("thick", "having much depth", "property"),
        DictionaryEntry("thin", "having little depth", "property"),
        DictionaryEntry("long", "having much length", "property"),
        DictionaryEntry("deep", "having much depth downward", "property"),
        DictionaryEntry("shallow", "having little depth", "property"),
        
        # State
        DictionaryEntry("empty", "containing nothing", "property"),
        DictionaryEntry("full", "containing everything possible", "property"),
        DictionaryEntry("open", "not closed", "property"),
        DictionaryEntry("closed", "not open", "property"),
        DictionaryEntry("broken", "not working correctly", "property"),
        DictionaryEntry("intact", "not broken", "property"),
        DictionaryEntry("complete", "having all parts", "property"),
        DictionaryEntry("incomplete", "missing some parts", "property"),
    ]


def create_extended_time_dictionary() -> List[DictionaryEntry]:
    """Extended time vocabulary."""
    return [
        # Duration
        DictionaryEntry("moment", "a very short time", "time"),
        DictionaryEntry("instant", "a very very short time", "time"),
        DictionaryEntry("second", "a short unit of time", "time"),
        DictionaryEntry("minute", "sixty seconds of time", "time"),
        DictionaryEntry("hour", "sixty minutes of time", "time"),
        DictionaryEntry("period", "an amount of time", "time"),
        DictionaryEntry("duration", "how long something lasts", "time"),
        DictionaryEntry("brief", "lasting a short time", "time"),
        DictionaryEntry("prolonged", "lasting a long time", "time"),
        
        # Sequence
        DictionaryEntry("before", "earlier in time", "time"),
        DictionaryEntry("after", "later in time", "time"),
        DictionaryEntry("during", "at the same time as", "time"),
        DictionaryEntry("while", "during the time that", "time"),
        DictionaryEntry("until", "up to the time that", "time"),
        DictionaryEntry("since", "from that time until now", "time"),
        DictionaryEntry("first", "before all others", "time"),
        DictionaryEntry("last", "after all others", "time"),
        DictionaryEntry("next", "coming after this", "time"),
        DictionaryEntry("previous", "coming before this", "time"),
        
        # Frequency
        DictionaryEntry("always", "at all times", "time"),
        DictionaryEntry("never", "at no time", "time"),
        DictionaryEntry("sometimes", "at some times", "time"),
        DictionaryEntry("often", "at many times", "time"),
        DictionaryEntry("rarely", "at few times", "time"),
        DictionaryEntry("usually", "most of the time", "time"),
        DictionaryEntry("occasionally", "sometimes but not often", "time"),
        DictionaryEntry("frequently", "at many times", "time"),
        DictionaryEntry("constantly", "without stopping", "time"),
        DictionaryEntry("intermittently", "starting and stopping", "time"),
        
        # Change
        DictionaryEntry("suddenly", "happening in an instant", "time"),
        DictionaryEntry("gradually", "happening slowly over time", "time"),
        DictionaryEntry("immediately", "right now without waiting", "time"),
        DictionaryEntry("eventually", "at some future time", "time"),
        DictionaryEntry("simultaneously", "at the same time", "time"),
        DictionaryEntry("sequentially", "one after another", "time"),
    ]


def create_extended_quantity_dictionary() -> List[DictionaryEntry]:
    """Extended quantity vocabulary."""
    return [
        # Numbers
        DictionaryEntry("zero", "the number of nothing", "quantity"),
        DictionaryEntry("one", "a single thing", "quantity"),
        DictionaryEntry("two", "one plus one", "quantity"),
        DictionaryEntry("three", "two plus one", "quantity"),
        DictionaryEntry("four", "three plus one", "quantity"),
        DictionaryEntry("five", "four plus one", "quantity"),
        DictionaryEntry("few", "a small number", "quantity"),
        DictionaryEntry("several", "more than two but not many", "quantity"),
        DictionaryEntry("many", "a large number", "quantity"),
        DictionaryEntry("numerous", "very many", "quantity"),
        DictionaryEntry("countless", "too many to count", "quantity"),
        
        # Comparison
        DictionaryEntry("more", "a larger amount", "quantity"),
        DictionaryEntry("less", "a smaller amount", "quantity"),
        DictionaryEntry("most", "the largest amount", "quantity"),
        DictionaryEntry("least", "the smallest amount", "quantity"),
        DictionaryEntry("equal", "the same amount", "quantity"),
        DictionaryEntry("unequal", "not the same amount", "quantity"),
        DictionaryEntry("double", "two times as much", "quantity"),
        DictionaryEntry("triple", "three times as much", "quantity"),
        DictionaryEntry("half", "one of two equal parts", "quantity"),
        DictionaryEntry("quarter", "one of four equal parts", "quantity"),
        
        # Completeness
        DictionaryEntry("all", "every one", "quantity"),
        DictionaryEntry("none", "not any", "quantity"),
        DictionaryEntry("some", "more than zero but not all", "quantity"),
        DictionaryEntry("any", "one or more of a group", "quantity"),
        DictionaryEntry("every", "each one in a group", "quantity"),
        DictionaryEntry("each", "every single one", "quantity"),
        DictionaryEntry("both", "the two together", "quantity"),
        DictionaryEntry("either", "one or the other", "quantity"),
        DictionaryEntry("neither", "not one and not the other", "quantity"),
        DictionaryEntry("whole", "all parts together", "quantity"),
        DictionaryEntry("partial", "only some parts", "quantity"),
        DictionaryEntry("entire", "all of something", "quantity"),
    ]


def create_full_extended_dictionary() -> List[DictionaryEntry]:
    """Create the complete extended dictionary."""
    all_entries = []
    all_entries.extend(create_extended_color_dictionary())
    all_entries.extend(create_extended_shape_dictionary())
    all_entries.extend(create_extended_spatial_dictionary())
    all_entries.extend(create_extended_action_dictionary())
    all_entries.extend(create_extended_property_dictionary())
    all_entries.extend(create_extended_time_dictionary())
    all_entries.extend(create_extended_quantity_dictionary())
    return all_entries


if __name__ == "__main__":
    dictionary = create_full_extended_dictionary()
    print(f"Extended dictionary: {len(dictionary)} entries")
    
    # Count by category
    categories = {}
    for entry in dictionary:
        cat = entry.category
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
