#!/usr/bin/env python3
"""
Vocabulary Grounding Pipeline

Connect 290k word embeddings to CHPL's visual concepts.

Strategy:
1. Direct grounding: Concrete nouns via ImageNet/COCO classes
2. Propagation: Abstract words via semantic neighbors
3. Validation: Test grounded words in visual QA

Target: 150k grounded words (50k direct + 100k propagated)
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ImageNet class names (1000 classes) - subset of most common
IMAGENET_CLASSES = [
    'tench', 'goldfish', 'shark', 'stingray', 'rooster', 'hen', 'ostrich', 'brambling',
    'goldfinch', 'house_finch', 'junco', 'indigo_bunting', 'robin', 'bulbul', 'jay',
    'magpie', 'chickadee', 'water_ouzel', 'kite', 'bald_eagle', 'vulture', 'great_grey_owl',
    'fire_salamander', 'newt', 'eft', 'spotted_salamander', 'axolotl', 'bullfrog', 'tree_frog',
    'tailed_frog', 'loggerhead', 'leatherback_turtle', 'mud_turtle', 'terrapin', 'box_turtle',
    'banded_gecko', 'iguana', 'alligator_lizard', 'agama', 'chameleon', 'whiptail', 'african_chameleon',
    'komodo_dragon', 'african_crocodile', 'american_alligator', 'triceratops', 'thunder_snake',
    'ringneck_snake', 'hognose_snake', 'green_snake', 'king_snake', 'garter_snake', 'water_snake',
    'vine_snake', 'night_snake', 'boa_constrictor', 'rock_python', 'indian_cobra', 'green_mamba',
    'sea_snake', 'horned_viper', 'diamondback', 'sidewinder', 'trilobite', 'harvestman', 'scorpion',
    'spider', 'tarantula', 'tick', 'centipede', 'black_grouse', 'ptarmigan', 'ruffed_grouse',
    'prairie_chicken', 'peacock', 'quail', 'partridge', 'african_grey', 'macaw', 'sulphur_crested_cockatoo',
    'lorikeet', 'coucal', 'bee_eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake',
    'red_breasted_merganser', 'goose', 'black_swan', 'tusker', 'echidna', 'platypus', 'wallaby',
    'koala', 'wombat', 'jellyfish', 'sea_anemone', 'brain_coral', 'flatworm', 'nematode', 'conch',
    'snail', 'slug', 'sea_slug', 'chiton', 'chambered_nautilus', 'dungeness_crab', 'rock_crab',
    'fiddler_crab', 'king_crab', 'american_lobster', 'spiny_lobster', 'crayfish', 'hermit_crab',
    'isopod', 'white_stork', 'black_stork', 'spoonbill', 'flamingo', 'little_blue_heron',
    'american_egret', 'bittern', 'crane', 'limpkin', 'european_gallinule', 'american_coot',
    'bustard', 'ruddy_turnstone', 'red_backed_sandpiper', 'redshank', 'dowitcher', 'oystercatcher',
    'pelican', 'king_penguin', 'albatross', 'grey_whale', 'killer_whale', 'dugong', 'sea_lion',
    'chihuahua', 'shih_tzu', 'afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick',
    'toy_terrier', 'walker_hound', 'english_foxhound', 'redbone', 'borzoi', 'irish_wolfhound',
    'italian_greyhound', 'whippet', 'ibizan_hound', 'norwegian_elkhound', 'otterhound', 'saluki',
    'scottish_deerhound', 'weimaraner', 'staffordshire_bullterrier', 'american_staffordshire_terrier',
    'bedlington_terrier', 'border_terrier', 'kerry_blue_terrier', 'irish_terrier', 'norfolk_terrier',
    'norwich_terrier', 'yorkshire_terrier', 'wire_haired_fox_terrier', 'lakeland_terrier',
    'sealyham_terrier', 'airedale', 'cairn', 'australian_terrier', 'dandie_dinmont', 'boston_bull',
    'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'scotch_terrier', 'tibetan_terrier',
    'silky_terrier', 'soft_coated_wheaten_terrier', 'west_highland_white_terrier', 'lhasa',
    'flat_coated_retriever', 'curly_coated_retriever', 'golden_retriever', 'labrador_retriever',
    'chesapeake_bay_retriever', 'german_short_haired_pointer', 'vizsla', 'english_setter',
    'irish_setter', 'gordon_setter', 'brittany_spaniel', 'clumber', 'english_springer',
    'welsh_springer_spaniel', 'cocker_spaniel', 'sussex_spaniel', 'irish_water_spaniel', 'kuvasz',
    'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'old_english_sheepdog',
    'shetland_sheepdog', 'collie', 'border_collie', 'bouvier_des_flandres', 'rottweiler',
    'german_shepherd', 'doberman', 'miniature_pinscher', 'greater_swiss_mountain_dog',
    'bernese_mountain_dog', 'appenzeller', 'entlebucher', 'boxer', 'bull_mastiff', 'tibetan_mastiff',
    'french_bulldog', 'great_dane', 'saint_bernard', 'eskimo_dog', 'malamute', 'siberian_husky',
    'dalmatian', 'affenpinscher', 'basenji', 'pug', 'leonberg', 'newfoundland', 'great_pyrenees',
    'samoyed', 'pomeranian', 'chow', 'keeshond', 'brabancon_griffon', 'pembroke', 'cardigan',
    'toy_poodle', 'miniature_poodle', 'standard_poodle', 'mexican_hairless', 'timber_wolf',
    'white_wolf', 'red_wolf', 'coyote', 'dingo', 'dhole', 'african_hunting_dog', 'hyena', 'red_fox',
    'kit_fox', 'arctic_fox', 'grey_fox', 'tabby', 'tiger_cat', 'persian_cat', 'siamese_cat',
    'egyptian_cat', 'cougar', 'lynx', 'leopard', 'snow_leopard', 'jaguar', 'lion', 'tiger', 'cheetah',
    'brown_bear', 'american_black_bear', 'ice_bear', 'sloth_bear', 'mongoose', 'meerkat', 'tiger_beetle',
    'ladybug', 'ground_beetle', 'long_horned_beetle', 'leaf_beetle', 'dung_beetle', 'rhinoceros_beetle',
    'weevil', 'fly', 'bee', 'ant', 'grasshopper', 'cricket', 'walking_stick', 'cockroach', 'mantis',
    'cicada', 'leafhopper', 'lacewing', 'dragonfly', 'damselfly', 'admiral', 'ringlet', 'monarch',
    'cabbage_butterfly', 'sulphur_butterfly', 'lycaenid', 'starfish', 'sea_urchin', 'sea_cucumber',
    'wood_rabbit', 'hare', 'angora', 'hamster', 'porcupine', 'fox_squirrel', 'marmot', 'beaver',
    'guinea_pig', 'sorrel', 'zebra', 'pig', 'wild_boar', 'warthog', 'hippopotamus', 'ox', 'water_buffalo',
    'bison', 'ram', 'bighorn', 'ibex', 'hartebeest', 'impala', 'gazelle', 'arabian_camel', 'llama',
    'weasel', 'mink', 'polecat', 'black_footed_ferret', 'otter', 'skunk', 'badger', 'armadillo',
    'three_toed_sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang', 'guenon', 'patas',
    'baboon', 'macaque', 'langur', 'colobus', 'proboscis_monkey', 'marmoset', 'capuchin', 'howler_monkey',
    'titi', 'spider_monkey', 'squirrel_monkey', 'madagascar_cat', 'indri', 'indian_elephant',
    'african_elephant', 'lesser_panda', 'giant_panda', 'barracouta', 'eel', 'coho', 'rock_beauty',
    'anemone_fish', 'sturgeon', 'gar', 'lionfish', 'puffer', 'abacus', 'abaya', 'academic_gown',
    'accordion', 'acoustic_guitar', 'aircraft_carrier', 'airliner', 'airship', 'altar', 'ambulance',
    'amphibian', 'analog_clock', 'apiary', 'apron', 'ashcan', 'assault_rifle', 'backpack', 'bakery',
    'balance_beam', 'balloon', 'ballpoint', 'band_aid', 'banjo', 'bannister', 'barbell', 'barber_chair',
    'barbershop', 'barn', 'barometer', 'barrel', 'barrow', 'baseball', 'basketball', 'bassinet',
    'bassoon', 'bathing_cap', 'bath_towel', 'bathtub', 'beach_wagon', 'beacon', 'beaker', 'bearskin',
    'beer_bottle', 'beer_glass', 'bell_cote', 'bib', 'bicycle', 'bikini', 'binder', 'binoculars',
    'birdhouse', 'boathouse', 'bobsled', 'bolo_tie', 'bonnet', 'bookcase', 'bookshop', 'bottlecap',
    'bow', 'bow_tie', 'brass', 'brassiere', 'breakwater', 'breastplate', 'broom', 'bucket', 'buckle',
    'bulletproof_vest', 'bullet_train', 'butcher_shop', 'cab', 'caldron', 'candle', 'cannon', 'canoe',
    'can_opener', 'cardigan', 'car_mirror', 'carousel', 'carpenters_kit', 'carton', 'car_wheel',
    'cash_machine', 'cassette', 'cassette_player', 'castle', 'catamaran', 'cd_player', 'cello',
    'cellular_telephone', 'chain', 'chainlink_fence', 'chain_mail', 'chain_saw', 'chest', 'chiffonier',
    'chime', 'china_cabinet', 'christmas_stocking', 'church', 'cinema', 'cleaver', 'cliff_dwelling',
    'cloak', 'clog', 'cocktail_shaker', 'coffee_mug', 'coffeepot', 'coil', 'combination_lock',
    'computer_keyboard', 'confectionery', 'container_ship', 'convertible', 'corkscrew', 'cornet',
    'cowboy_boot', 'cowboy_hat', 'cradle', 'crane', 'crash_helmet', 'crate', 'crib', 'crock_pot',
    'croquet_ball', 'crutch', 'cuirass', 'dam', 'desk', 'desktop_computer', 'dial_telephone',
    'diaper', 'digital_clock', 'digital_watch', 'dining_table', 'dishrag', 'dishwasher', 'disk_brake',
    'dock', 'dogsled', 'dome', 'doormat', 'drilling_platform', 'drum', 'drumstick', 'dumbbell',
    'dutch_oven', 'electric_fan', 'electric_guitar', 'electric_locomotive', 'entertainment_center',
    'envelope', 'espresso_maker', 'face_powder', 'feather_boa', 'file', 'fireboat', 'fire_engine',
    'fire_screen', 'flagpole', 'flute', 'folding_chair', 'football_helmet', 'forklift', 'fountain',
    'fountain_pen', 'four_poster', 'freight_car', 'french_horn', 'frying_pan', 'fur_coat', 'garbage_truck',
    'gasmask', 'gas_pump', 'goblet', 'go_kart', 'golf_ball', 'golfcart', 'gondola', 'gong', 'gown',
    'grand_piano', 'greenhouse', 'grille', 'grocery_store', 'guillotine', 'hair_slide', 'hair_spray',
    'half_track', 'hammer', 'hamper', 'hand_blower', 'hand_held_computer', 'handkerchief', 'hard_disc',
    'harmonica', 'harp', 'harvester', 'hatchet', 'holster', 'home_theater', 'honeycomb', 'hook',
    'hoopskirt', 'horizontal_bar', 'horse_cart', 'hourglass', 'ipod', 'iron', 'jack_o_lantern', 'jean',
    'jeep', 'jersey', 'jigsaw_puzzle', 'jinrikisha', 'joystick', 'kimono', 'knee_pad', 'knot',
    'lab_coat', 'ladle', 'lampshade', 'laptop', 'lawn_mower', 'lens_cap', 'letter_opener', 'library',
    'lifeboat', 'lighter', 'limousine', 'liner', 'lipstick', 'loafer', 'lotion', 'loudspeaker',
    'loupe', 'lumbermill', 'magnetic_compass', 'mailbag', 'mailbox', 'maillot', 'manhole_cover',
    'maraca', 'marimba', 'mask', 'matchstick', 'maypole', 'maze', 'measuring_cup', 'medicine_chest',
    'megalith', 'microphone', 'microwave', 'military_uniform', 'milk_can', 'minibus', 'miniskirt',
    'minivan', 'missile', 'mitten', 'mixing_bowl', 'mobile_home', 'model_t', 'modem', 'monastery',
    'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito_net', 'motor_scooter', 'mountain_bike',
    'mountain_tent', 'mouse', 'mousetrap', 'moving_van', 'muzzle', 'nail', 'neck_brace', 'necklace',
    'nipple', 'notebook', 'obelisk', 'oboe', 'ocarina', 'odometer', 'oil_filter', 'organ', 'oscilloscope',
    'overskirt', 'oxcart', 'oxygen_mask', 'packet', 'paddle', 'paddlewheel', 'padlock', 'paintbrush',
    'pajama', 'palace', 'panpipe', 'paper_towel', 'parachute', 'parallel_bars', 'park_bench', 'parking_meter',
    'passenger_car', 'patio', 'pay_phone', 'pedestal', 'pencil_box', 'pencil_sharpener', 'perfume',
    'petri_dish', 'photocopier', 'pick', 'pickelhaube', 'picket_fence', 'pickup', 'pier', 'piggy_bank',
    'pill_bottle', 'pillow', 'ping_pong_ball', 'pinwheel', 'pirate', 'pitcher', 'plane', 'planetarium',
    'plastic_bag', 'plate_rack', 'plow', 'plunger', 'polaroid_camera', 'pole', 'police_van', 'poncho',
    'pool_table', 'pop_bottle', 'pot', 'potters_wheel', 'power_drill', 'prayer_rug', 'printer',
    'prison', 'projectile', 'projector', 'puck', 'punching_bag', 'purse', 'quill', 'quilt', 'racer',
    'racket', 'radiator', 'radio', 'radio_telescope', 'rain_barrel', 'recreational_vehicle', 'reel',
    'reflex_camera', 'refrigerator', 'remote_control', 'restaurant', 'revolver', 'rifle', 'rocking_chair',
    'rotisserie', 'rubber_eraser', 'rugby_ball', 'rule', 'running_shoe', 'safe', 'safety_pin', 'saltshaker',
    'sandal', 'sarong', 'sax', 'scabbard', 'scale', 'school_bus', 'schooner', 'scoreboard', 'screen',
    'screw', 'screwdriver', 'seat_belt', 'sewing_machine', 'shield', 'shoe_shop', 'shoji', 'shopping_basket',
    'shopping_cart', 'shovel', 'shower_cap', 'shower_curtain', 'ski', 'ski_mask', 'sleeping_bag',
    'slide_rule', 'sliding_door', 'slot', 'snorkel', 'snowmobile', 'snowplow', 'soap_dispenser',
    'soccer_ball', 'sock', 'solar_dish', 'sombrero', 'soup_bowl', 'space_bar', 'space_heater',
    'space_shuttle', 'spatula', 'speedboat', 'spider_web', 'spindle', 'sports_car', 'spotlight',
    'stage', 'steam_locomotive', 'steel_arch_bridge', 'steel_drum', 'stethoscope', 'stole', 'stone_wall',
    'stopwatch', 'stove', 'strainer', 'streetcar', 'stretcher', 'studio_couch', 'stupa', 'submarine',
    'suit', 'sundial', 'sunglass', 'sunglasses', 'sunscreen', 'suspension_bridge', 'swab', 'sweatshirt',
    'swimming_trunks', 'swing', 'switch', 'syringe', 'table_lamp', 'tank', 'tape_player', 'teapot',
    'teddy', 'television', 'tennis_ball', 'thatch', 'theater_curtain', 'thimble', 'thresher', 'throne',
    'tile_roof', 'toaster', 'tobacco_shop', 'toilet_seat', 'torch', 'totem_pole', 'tow_truck', 'toyshop',
    'tractor', 'trailer_truck', 'tray', 'trench_coat', 'tricycle', 'trimaran', 'tripod', 'triumphal_arch',
    'trolleybus', 'trombone', 'tub', 'turnstile', 'typewriter_keyboard', 'umbrella', 'unicycle', 'upright',
    'vacuum', 'vase', 'vault', 'velvet', 'vending_machine', 'vestment', 'viaduct', 'violin', 'volleyball',
    'waffle_iron', 'wall_clock', 'wallet', 'wardrobe', 'warplane', 'washbasin', 'washer', 'water_bottle',
    'water_jug', 'water_tower', 'whiskey_jug', 'whistle', 'wig', 'window_screen', 'window_shade',
    'windsor_tie', 'wine_bottle', 'wing', 'wok', 'wooden_spoon', 'wool', 'worm_fence', 'wreck', 'yawl',
    'yurt', 'web_site', 'comic_book', 'crossword_puzzle', 'street_sign', 'traffic_light', 'book_jacket',
    'menu', 'plate', 'guacamole', 'consomme', 'hot_pot', 'trifle', 'ice_cream', 'ice_lolly', 'french_loaf',
    'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'mashed_potato', 'head_cabbage', 'broccoli', 'cauliflower',
    'zucchini', 'spaghetti_squash', 'acorn_squash', 'butternut_squash', 'cucumber', 'artichoke', 'bell_pepper',
    'cardoon', 'mushroom', 'granny_smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana',
    'jackfruit', 'custard_apple', 'pomegranate', 'hay', 'carbonara', 'chocolate_sauce', 'dough', 'meat_loaf',
    'pizza', 'potpie', 'burrito', 'red_wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble', 'cliff',
    'coral_reef', 'geyser', 'lakeside', 'promontory', 'sandbar', 'seashore', 'valley', 'volcano', 'ballplayer',
    'groom', 'scuba_diver', 'rapeseed', 'daisy', 'yellow_lady_slipper', 'corn', 'acorn', 'hip', 'buckeye',
    'coral_fungus', 'agaric', 'gyromitra', 'stinkhorn', 'earthstar', 'hen_of_the_woods', 'bolete', 'ear',
    'toilet_tissue'
]

# Common concrete nouns (basic vocabulary)
CONCRETE_NOUNS = [
    # Colors
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray', 'grey',
    # Shapes
    'circle', 'square', 'triangle', 'rectangle', 'oval', 'diamond', 'star', 'heart',
    # Sizes
    'big', 'small', 'large', 'tiny', 'huge', 'little', 'medium',
    # Basic objects
    'ball', 'box', 'cup', 'plate', 'book', 'pen', 'paper', 'table', 'chair', 'door', 'window',
    'car', 'bus', 'train', 'plane', 'boat', 'bicycle', 'truck',
    'tree', 'flower', 'grass', 'leaf', 'sun', 'moon', 'star', 'cloud', 'rain', 'snow',
    'house', 'building', 'road', 'bridge', 'mountain', 'river', 'lake', 'ocean', 'beach',
    'person', 'man', 'woman', 'child', 'boy', 'girl', 'baby', 'face', 'hand', 'eye', 'head',
    'dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'pig', 'sheep', 'chicken', 'duck',
    'apple', 'banana', 'orange', 'bread', 'milk', 'water', 'food', 'meat', 'vegetable', 'fruit',
    # Actions (can be visually grounded)
    'run', 'walk', 'jump', 'sit', 'stand', 'fall', 'fly', 'swim', 'eat', 'drink', 'sleep',
    # Spatial
    'up', 'down', 'left', 'right', 'top', 'bottom', 'inside', 'outside', 'above', 'below',
]


@dataclass
class GroundedWord:
    """A word with visual grounding."""
    word: str
    activation: torch.Tensor
    grounding_type: str  # 'direct', 'propagated', 'synthetic'
    confidence: float
    source: str  # What provided the grounding


class VocabularyGrounder:
    """
    Connect word embeddings to CHPL's visual concepts.
    
    Three-stage grounding:
    1. Direct: Words with visual examples (ImageNet/COCO classes)
    2. Propagated: Words reachable via semantic neighbors
    3. Synthetic: Words grounded via generated images
    """
    
    def __init__(
        self,
        word2vec_model,
        chpl_brain,
        embedding_dim: int = 64,
    ):
        self.word2vec = word2vec_model
        self.brain = chpl_brain
        self.embedding_dim = embedding_dim
        
        # Grounded words
        self.grounded: Dict[str, GroundedWord] = {}
        
        # Statistics
        self.stats = {
            'direct': 0,
            'propagated': 0,
            'synthetic': 0,
            'total': 0,
        }
    
    def ground_imagenet_classes(self, imagenet_path: Optional[str] = None) -> int:
        """
        Ground words that correspond to ImageNet classes.
        
        If imagenet_path is provided, use actual images.
        Otherwise, use synthetic activations based on class semantics.
        """
        print("\n" + "=" * 60)
        print("Stage 1: Grounding ImageNet classes")
        print("=" * 60)
        
        grounded_count = 0
        
        for class_name in IMAGENET_CLASSES:
            # Normalize class name
            word = class_name.lower().replace('_', ' ')
            words = word.split()
            
            # Check if any word form is in vocabulary
            word_in_vocab = None
            for w in words + [class_name.lower(), class_name.replace('_', '')]:
                if w in self.word2vec.vocab:
                    word_in_vocab = w
                    break
            
            if word_in_vocab is None:
                continue
            
            # Get word embedding from embeddings tensor
            word_idx = self.word2vec.vocab[word_in_vocab]
            word_emb = self.word2vec.embeddings[word_idx]
            
            if word_emb is None:
                continue
            
            # Create visual activation (synthetic if no images)
            if imagenet_path and Path(imagenet_path).exists():
                # Load actual images and compute activation
                activation = self._compute_activation_from_images(
                    imagenet_path, class_name
                )
            else:
                # Create synthetic activation based on word embedding
                activation = self._create_synthetic_activation(word_emb)
            
            # Store grounded word
            self.grounded[word_in_vocab] = GroundedWord(
                word=word_in_vocab,
                activation=activation,
                grounding_type='direct',
                confidence=0.9,
                source=f'imagenet:{class_name}',
            )
            
            grounded_count += 1
        
        self.stats['direct'] = grounded_count
        print(f"  Grounded {grounded_count} ImageNet classes")
        
        return grounded_count
    
    def ground_coco_captions(self, coco_path: str, max_images: int = 50000) -> int:
        """
        Ground words from COCO image captions using real visual activations.
        
        For each image:
        1. Compute CHPL visual activation
        2. Extract nouns/objects from caption
        3. Link caption words to visual activation
        """
        import cv2
        import re
        
        print("\n" + "=" * 60)
        print(f"Stage: Grounding from COCO captions ({max_images} images)")
        print("=" * 60)
        
        coco_dir = Path(coco_path)
        annotations_file = coco_dir.parent / 'annotations' / 'captions_train2017.json'
        
        if not annotations_file.exists():
            print(f"  COCO annotations not found: {annotations_file}")
            return 0
        
        # Load COCO captions
        print("  Loading COCO annotations...")
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image_id -> captions mapping
        image_captions = defaultdict(list)
        for ann in coco_data['annotations']:
            image_captions[ann['image_id']].append(ann['caption'])
        
        # Build image_id -> filename mapping
        image_files = {img['id']: img['file_name'] for img in coco_data['images']}
        
        print(f"  Found {len(image_files)} images with {len(coco_data['annotations'])} captions")
        
        # Process images
        grounded_count = 0
        processed = 0
        word_activations = defaultdict(list)  # word -> list of activations
        
        # Stopwords to filter out
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then', 'once',
                     'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'each', 'few', 'more', 'most', 'other', 'some', 'such',
                     'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                     'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
                     'until', 'while', 'that', 'this', 'these', 'those', 'what',
                     'which', 'who', 'whom', 'its', 'it', 'he', 'she', 'they',
                     'their', 'his', 'her', 'my', 'your', 'our', 'i', 'you', 'we'}
        
        image_ids = list(image_files.keys())[:max_images]
        
        for i, img_id in enumerate(image_ids):
            if i % 5000 == 0:
                print(f"    Processed {i}/{len(image_ids)} images, {len(word_activations)} unique words...")
            
            img_file = image_files[img_id]
            img_path = coco_dir / img_file
            
            if not img_path.exists():
                continue
            
            try:
                # Load and process image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (56, 56))
                
                img_t = torch.from_numpy(img).float() / 255.0
                img_t = img_t.permute(2, 0, 1).to(DEVICE)
                
                # Get CHPL visual activation
                with torch.no_grad():
                    result = self.brain.visual(img_t.unsqueeze(0))
                    # Handle both tuple and single tensor returns
                    if isinstance(result, tuple):
                        activation = result[0].squeeze(0)
                    else:
                        activation = result.squeeze(0) if result.dim() > 1 else result
                
                # Extract words from captions
                for caption in image_captions[img_id]:
                    # Tokenize caption
                    words = re.findall(r'\b[a-z]+\b', caption.lower())
                    
                    for word in words:
                        # Skip stopwords and short words
                        if word in stopwords or len(word) < 3:
                            continue
                        
                        # Check if word is in vocabulary
                        if word not in self.word2vec.vocab:
                            continue
                        
                        # Skip if already grounded with high confidence
                        if word in self.grounded and self.grounded[word].confidence > 0.9:
                            continue
                        
                        # Accumulate activation for this word
                        word_activations[word].append(activation.clone())
                
                processed += 1
                
            except Exception as e:
                continue
        
        print(f"  Processed {processed} images")
        print(f"  Computing average activations for {len(word_activations)} words...")
        
        # Average activations for each word
        for word, acts in word_activations.items():
            if len(acts) >= 3:  # Require at least 3 examples
                avg_activation = torch.stack(acts).mean(dim=0)
                avg_activation = F.normalize(avg_activation.unsqueeze(0), dim=1).squeeze(0)
                
                # Confidence based on number of examples
                confidence = min(0.95, 0.5 + 0.05 * len(acts))
                
                self.grounded[word] = GroundedWord(
                    word=word,
                    activation=avg_activation,
                    grounding_type='direct',
                    confidence=confidence,
                    source=f'coco:{len(acts)}_examples',
                )
                grounded_count += 1
        
        self.stats['direct'] += grounded_count
        print(f"  Grounded {grounded_count} words from COCO captions")
        
        return grounded_count
    
    def ground_concrete_nouns(self) -> int:
        """Ground basic concrete nouns with synthetic activations."""
        print("\n" + "=" * 60)
        print("Stage 2: Grounding concrete nouns")
        print("=" * 60)
        
        grounded_count = 0
        
        for word in CONCRETE_NOUNS:
            if word in self.grounded:
                continue
            
            # Check if in vocabulary
            if word not in self.word2vec.vocab:
                continue
            
            # Get word embedding
            word_idx = self.word2vec.vocab[word]
            word_emb = self.word2vec.embeddings[word_idx]
            
            if word_emb is None:
                continue
            
            # Create synthetic activation
            activation = self._create_synthetic_activation(word_emb)
            
            self.grounded[word] = GroundedWord(
                word=word,
                activation=activation,
                grounding_type='direct',
                confidence=0.85,
                source='concrete_noun',
            )
            
            grounded_count += 1
        
        self.stats['direct'] += grounded_count
        print(f"  Grounded {grounded_count} concrete nouns")
        
        return grounded_count
    
    def propagate_grounding(self, max_hops: int = 2, min_neighbors: int = 3, max_words: int = 50000) -> int:
        """
        Propagate grounding to abstract words via semantic neighbors.
        
        For each ungrounded word:
        1. Find its grounded neighbors in word2vec space
        2. Weight their activations by similarity
        3. Store weighted average as grounding
        
        Optimized: Only process most frequent words first.
        """
        print("\n" + "=" * 60)
        print(f"Stage 3: Propagating grounding (max {max_hops} hops, {max_words} words)")
        print("=" * 60)
        
        total_propagated = 0
        
        # Sort words by frequency (most common first)
        if hasattr(self.word2vec, 'word_counts'):
            sorted_words = sorted(
                self.word2vec.vocab.keys(),
                key=lambda w: self.word2vec.word_counts.get(w, 0),
                reverse=True
            )[:max_words]
        else:
            sorted_words = list(self.word2vec.vocab.keys())[:max_words]
        
        for hop in range(max_hops):
            print(f"\n  Hop {hop + 1}...")
            
            propagated_this_hop = 0
            grounded_words = set(self.grounded.keys())
            
            for i, word in enumerate(sorted_words):
                if i % 10000 == 0 and i > 0:
                    print(f"    Processed {i}/{len(sorted_words)}, propagated {propagated_this_hop}...")
                
                if word in self.grounded:
                    continue
                
                # Find similar grounded words
                try:
                    similar = self.word2vec.get_similar_words(word, 20)
                except:
                    continue
                
                # Filter to grounded neighbors
                grounded_neighbors = [
                    (w, sim) for w, sim in similar
                    if w in grounded_words
                ]
                
                if len(grounded_neighbors) < min_neighbors:
                    continue
                
                # Weighted average of neighbor activations
                weights = torch.tensor([sim for _, sim in grounded_neighbors[:10]]).to(DEVICE)
                weights = F.softmax(weights * 2, dim=0)  # Temperature scaling
                
                neighbor_acts = torch.stack([
                    self.grounded[w].activation.to(DEVICE)
                    for w, _ in grounded_neighbors[:10]
                ]).to(DEVICE)
                
                activation = (neighbor_acts * weights.unsqueeze(1)).sum(dim=0)
                activation = F.normalize(activation.unsqueeze(0), dim=1).squeeze(0)
                
                # Confidence based on neighbor similarity
                avg_sim = sum(sim for _, sim in grounded_neighbors[:10]) / 10
                confidence = avg_sim * (0.8 ** (hop + 1))  # Decay with hops
                
                self.grounded[word] = GroundedWord(
                    word=word,
                    activation=activation,
                    grounding_type='propagated',
                    confidence=confidence,
                    source=f'propagated_hop{hop + 1}',
                )
                
                propagated_this_hop += 1
            
            print(f"    Propagated {propagated_this_hop} words")
            total_propagated += propagated_this_hop
            
            if propagated_this_hop == 0:
                break
        
        self.stats['propagated'] = total_propagated
        print(f"\n  Total propagated: {total_propagated}")
        
        return total_propagated
    
    def _compute_activation_from_images(
        self,
        imagenet_path: str,
        class_name: str,
        n_samples: int = 10,
    ) -> torch.Tensor:
        """Compute activation from actual images."""
        import cv2
        
        class_dir = Path(imagenet_path) / class_name
        
        if not class_dir.exists():
            return self._create_synthetic_activation(None)
        
        activations = []
        
        for img_path in list(class_dir.glob('*.JPEG'))[:n_samples]:
            try:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (56, 56))
                
                img_t = torch.from_numpy(img).float() / 255.0
                img_t = img_t.permute(2, 0, 1).to(DEVICE)
                
                with torch.no_grad():
                    result = self.brain.visual(img_t.unsqueeze(0))
                    if isinstance(result, tuple):
                        features = result[0].squeeze(0)
                    else:
                        features = result.squeeze(0) if result.dim() > 1 else result
                    activations.append(features)
            except:
                continue
        
        if activations:
            return torch.stack(activations).mean(dim=0)
        else:
            return self._create_synthetic_activation(None)
    
    def _create_synthetic_activation(
        self,
        word_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Create synthetic visual activation from word embedding."""
        if word_embedding is not None:
            # Project word embedding to visual space
            # Use a learned or random projection
            if word_embedding.shape[0] != self.embedding_dim:
                # Simple linear projection
                proj = torch.randn(word_embedding.shape[0], self.embedding_dim)
                proj = proj / proj.norm(dim=0, keepdim=True)
                activation = word_embedding @ proj
            else:
                activation = word_embedding.clone()
            
            activation = F.normalize(activation.unsqueeze(0), dim=1).squeeze(0)
        else:
            # Random activation
            activation = torch.randn(self.embedding_dim)
            activation = F.normalize(activation.unsqueeze(0), dim=1).squeeze(0)
        
        return activation.to(DEVICE)
    
    def get_grounding(self, word: str) -> Optional[GroundedWord]:
        """Get grounding for a word."""
        return self.grounded.get(word)
    
    def is_grounded(self, word: str) -> bool:
        """Check if word is grounded."""
        return word in self.grounded
    
    def get_stats(self) -> Dict:
        """Get grounding statistics."""
        self.stats['total'] = len(self.grounded)
        
        # Count by type
        by_type = defaultdict(int)
        by_confidence = {'high': 0, 'medium': 0, 'low': 0}
        
        for gw in self.grounded.values():
            by_type[gw.grounding_type] += 1
            
            if gw.confidence > 0.7:
                by_confidence['high'] += 1
            elif gw.confidence > 0.4:
                by_confidence['medium'] += 1
            else:
                by_confidence['low'] += 1
        
        return {
            'total': self.stats['total'],
            'by_type': dict(by_type),
            'by_confidence': by_confidence,
        }
    
    def save(self, path: str):
        """Save grounded vocabulary."""
        data = {
            'grounded': {
                word: {
                    'word': gw.word,
                    'activation': gw.activation.cpu().numpy(),
                    'grounding_type': gw.grounding_type,
                    'confidence': gw.confidence,
                    'source': gw.source,
                }
                for word, gw in self.grounded.items()
            },
            'stats': self.stats,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(self.grounded)} grounded words to {path}")
    
    def load(self, path: str):
        """Load grounded vocabulary."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.grounded = {}
        for word, gw_data in data['grounded'].items():
            self.grounded[word] = GroundedWord(
                word=gw_data['word'],
                activation=torch.from_numpy(gw_data['activation']).to(DEVICE),
                grounding_type=gw_data['grounding_type'],
                confidence=gw_data['confidence'],
                source=gw_data['source'],
            )
        
        self.stats = data['stats']
        print(f"Loaded {len(self.grounded)} grounded words from {path}")


def run_grounding_experiment(
    word2vec_path: str,
    brain_path: Optional[str] = None,
    imagenet_path: Optional[str] = None,
    coco_path: Optional[str] = None,
    output_path: str = 'grounded_vocabulary.pkl',
    max_coco_images: int = 50000,
):
    """Run the complete grounding experiment."""
    print("=" * 70)
    print("VOCABULARY GROUNDING EXPERIMENT")
    print("Connecting words to visual concepts")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Load word2vec model
    print("\nLoading Word2Vec model...")
    from distributional_language import DistributionalLanguage
    
    dl = DistributionalLanguage()
    dl.load(word2vec_path)
    
    vocab_size = len(dl.vocab) if hasattr(dl, 'vocab') else len(dl.model.wv)
    print(f"  Loaded vocabulary: {vocab_size:,} words")
    
    # Load or create brain
    print("\nLoading CHPL brain...")
    
    if brain_path and Path(brain_path).exists():
        from hierarchical_atl import AbstractBrain
        
        brain = AbstractBrain(feature_dim=64, n_concepts=200, visual_input_size=56)
        checkpoint = torch.load(brain_path, map_location=DEVICE, weights_only=True)
        brain.visual.load_state_dict(checkpoint['visual_state'])
        print(f"  Loaded brain from {brain_path}")
    else:
        # Create minimal brain for activation generation
        brain = None
        print("  No brain loaded (using synthetic activations)")
    
    # Create grounder
    grounder = VocabularyGrounder(dl, brain, embedding_dim=64)
    
    # Stage 1: COCO captions (if available)
    if coco_path and Path(coco_path).exists():
        grounder.ground_coco_captions(coco_path, max_images=max_coco_images)
    
    # Stage 2: ImageNet classes
    grounder.ground_imagenet_classes(imagenet_path)
    
    # Stage 3: Concrete nouns
    grounder.ground_concrete_nouns()
    
    # Stage 4: Propagate (with more words now that we have COCO)
    grounder.propagate_grounding(max_hops=3, min_neighbors=2, max_words=100000)
    
    # Statistics
    stats = grounder.get_stats()
    
    print("\n" + "=" * 60)
    print("GROUNDING COMPLETE")
    print("=" * 60)
    
    print(f"\n  Total grounded: {stats['total']:,} words")
    print(f"  By type: {stats['by_type']}")
    print(f"  By confidence: {stats['by_confidence']}")
    
    grounding_rate = stats['total'] / vocab_size * 100
    print(f"\n  Grounding rate: {grounding_rate:.1f}%")
    
    # Save
    grounder.save(output_path)
    
    # Duration
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n  Duration: {duration:.1f} seconds")
    
    return grounder, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec', type=str, required=True,
                       help='Path to Word2Vec model')
    parser.add_argument('--brain', type=str, default=None,
                       help='Path to CHPL brain checkpoint')
    parser.add_argument('--imagenet', type=str, default=None,
                       help='Path to ImageNet images')
    parser.add_argument('--coco', type=str, default=None,
                       help='Path to COCO train2017 images')
    parser.add_argument('--max-coco', type=int, default=50000,
                       help='Max COCO images to process')
    parser.add_argument('--output', type=str, default='grounded_vocabulary.pkl',
                       help='Output path')
    
    args = parser.parse_args()
    
    run_grounding_experiment(
        word2vec_path=args.word2vec,
        brain_path=args.brain,
        imagenet_path=args.imagenet,
        coco_path=args.coco,
        output_path=args.output,
        max_coco_images=args.max_coco,
    )
