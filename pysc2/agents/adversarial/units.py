# data not provided in pysc2 so needs to be taken from online wikis for starcraft 2
# not complete resource
# needs to be updated according to units being used

# for convenience
units = {
    "HELLION": 53,
    "MARINE": 48,
    "STALKER": 74,
    "ROACH": 110,
    "ZEALOT": 73,
    "REAPER": 49,
    "ADEPT": 311,
    "VOIDRAY": 80,
    "ZERGLING": 105,
    "ULTRALISK": 109
}

weapon_ranges = {
    units["HELLION"]: 5,
    units["MARINE"]: 5,
    units["STALKER"]: 6,
    units["ROACH"]: 4,
    units["ZEALOT"]: 1,
    units["REAPER"]: 5,
    units["ADEPT"]: 4,
    units["VOIDRAY"]: 6,
    units["ZERGLING"]: 0.1,
    units["ULTRALISK"]: 1
}

# pysc2 provides radius as zero for some reason
unit_sizes = {
    units["HELLION"]: 1.25,
    units["MARINE"]: 0.75,
    units["STALKER"]: 1.25,
    units["ROACH"]: 1,
    units["ZEALOT"]: 1,
    units["REAPER"]: 0.75,
    units["ADEPT"]: 1,
    units["VOIDRAY"]: 2,
    units["ZERGLING"]: 0.75,
    units["ULTRALISK"]: 2
}

# whether unit is ranged 1 or melee 0
unit_type = {
    units["HELLION"]: 1,
    units["MARINE"]: 1,
    units["STALKER"]: 1,
    units["ROACH"]: 1,
    units["ZEALOT"]: 0,
    units["REAPER"]: 1,
    units["ADEPT"]: 1,
    units["VOIDRAY"]: 1,
    units["ZERGLING"]: 0,
    units["ULTRALISK"]: 0
}

# unit movement speed
unit_speed = {
    units["HELLION"]: 5.95,
    units["MARINE"]: 3.15,
    units["STALKER"]: 4.13,
    units["ROACH"]: 3.15,
    units["ZEALOT"]: 3.15,
    units["REAPER"]: 5.25,
    units["ADEPT"]: 3.5,
    units["VOIDRAY"]: 3.5,
    units["ZERGLING"]: 4.13,
    units["ULTRALISK"]: 4.13
}

unit_health = {
    units["HELLION"]: 90,
    units["STALKER"]: 80
}
