from Enums import HatchLevel, SpeciesMultiplier, ChargeMultiplier
from Constants import DATA_DIR
import pandas as pd

MIN_LEVEL = 1
MAX_LEVEL = 120
MIN_CLONE = 0
MAX_CLONE = 60

def getRaremonExperience(hatchLevel: HatchLevel, chargeMultiplier = ChargeMultiplier.REGULAR_CHARGE):
    raremonExperiences = {
        HatchLevel.LEVEL3: 2.24,
        HatchLevel.LEVEL4: 16.82,
        HatchLevel.LEVEL5: 44.85,
    }
    return raremonExperiences[hatchLevel] * chargeMultiplier.value

def calcExperience(level: int, clone: int, hatchLevel: HatchLevel, chargeMultiplier = ChargeMultiplier.REGULAR_CHARGE, speciesMultiplier = SpeciesMultiplier.DIFFERENT_SPECIES):
    if level < MIN_LEVEL or level > MAX_LEVEL:
        raise Exception(f"Invalid level range, range should be {MIN_LEVEL}~{MAX_LEVEL} inclusive")
    
    if clone < MIN_CLONE or clone > MAX_CLONE:
        raise Exception(f"Invalid clone range, clone should be {MIN_CLONE}~{MAX_CLONE} inclusive")
    
    return (clone * 5 + level + 250) * (hatchLevel.value * speciesMultiplier.value) * chargeMultiplier.value / 1400

def genData(partial: bool):
    genType = "partial" if partial else "full"
    file_name = f"{DATA_DIR}\\{genType}_data_set.txt"
    content = ""

    print(f"{genType} data generation started")

    for hatch in HatchLevel:
        for charge in ChargeMultiplier:
            for species in SpeciesMultiplier:
                for level in range(MIN_LEVEL, (int(MAX_LEVEL / 2) if partial else MAX_LEVEL) + 1):
                    for clone in range(MIN_CLONE, (int(MAX_CLONE / 2) if partial else MAX_CLONE) + 1):
                        experience = round(calcExperience(level, clone, hatch, charge, species), 2)
                        content += f"Hatch: {hatch.name} | Charge: {charge.name} | Species: {species.name} | Level: {level} | Clone: {clone} | Experience: {experience}\n"

    with open(file_name, 'w') as file:
        file.write(content)

    print(f"{genType} data generated at " + file_name)

def genDataForNeuralEducator(partial: bool):
    genType = "partial" if partial else "full"
    file_name = f"{DATA_DIR}\\{genType}_neural_educator_data_set.txt"
    content = f"Nome\nDMTNN\nEntradas\n\n\n\n\n\nSaidas\n\n"

    print(f"{genType} data generation for neural educator started")

    # 3,4,5,6,7 -> Entradas | 9 -> Saidas
    lines = content.splitlines()

    for hatch in HatchLevel:
        for charge in ChargeMultiplier:
            for species in SpeciesMultiplier:
                for level in range(MIN_LEVEL, (int(MAX_LEVEL / 2) if partial else MAX_LEVEL) + 1):
                    for clone in range(MIN_CLONE, (int(MAX_CLONE / 2) if partial else MAX_CLONE) + 1):
                        lines[3] += f"{hatch.value},"
                        lines[4] += f"{charge.value},"
                        lines[5] += f"{species.value},"
                        lines[6] += f"{level},"
                        lines[7] += f"{clone},"
                        lines[9] += f"{round(calcExperience(level, clone, hatch, charge, species), 2)},"

    lines[3] = lines[3][:-1]
    lines[4] = lines[4][:-1]
    lines[5] = lines[5][:-1]
    lines[6] = lines[6][:-1]
    lines[7] = lines[7][:-1]
    lines[9] = lines[9][:-1]

    finalContent = ""
    for i in range(len(lines)):
        finalContent += lines[i] + "\n"

    with open(file_name, 'w') as file:
        file.write(finalContent)

    print(f"{genType} data generated for neural educator at " + file_name)

def readData(partial: bool):
    genType = "partial" if partial else "full"
    file_name = f"{DATA_DIR}\\{genType}_data_set.txt"
    file = open(file_name)
    allLines = file.readlines()
    
    hatches, charges, species, levels, clones, experiences = [],[],[],[],[],[]

    for i in range(len(allLines)):
        status = allLines[i].split("|")
        hatches.append(HatchLevel[(status[0].split(":")[1]).strip()].value)
        charges.append(ChargeMultiplier[(status[1].split(":")[1]).strip()].value)
        species.append(SpeciesMultiplier[(status[2].split(":")[1]).strip()].value)
        levels.append(int(status[3].split(":")[1]))
        clones.append(int(status[4].split(":")[1]))
        experiences.append(float(status[5].split(":")[1]))

    dataSet = pd.DataFrame({
        'Hatch': hatches,
        'Charge': charges,
        'Species': species,
        'Level': levels,
        'Clone': clones,
        'Experience': experiences
    })

    return dataSet

# genData(partial=True)
# genData(partial=False)

# genDataForNeuralEducator(partial=True)
# genDataForNeuralEducator(partial=False)

# readData(partial=True)
# readData(partial=False)