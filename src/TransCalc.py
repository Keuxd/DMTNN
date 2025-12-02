from Enums import HatchLevel, SpeciesMultiplier, ChargeMultiplier
import pandas as pd

DATA_DIR = "D:\\DMTNN\\"

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
        raise Exception("Invalid level range, range should be 1~120 inclusive")
    
    if clone < MIN_CLONE or clone > MAX_CLONE:
        raise Exception("Invalid clone range, clone should be 0~60 inclusive")
    
    return (clone * 5 + level + 250) * (hatchLevel.value * speciesMultiplier.value) * chargeMultiplier.value / 1400

def genDataForNeuralEducator(hatchLevel: HatchLevel, chargeMultiplier: ChargeMultiplier, speciesMultiplier: SpeciesMultiplier):
    file_name = f"{DATA_DIR}DMTNN-{hatchLevel.name}-{chargeMultiplier.name}-{speciesMultiplier.name}.txt"
    content = f"Nome\nDMTNN-{hatchLevel.name}-{chargeMultiplier.name}-{speciesMultiplier.name}\nEntradas\n\n\nSaidas\n\n"

    # 3,4 -> Entradas | 6 -> Saidas
    lines = content.splitlines()

    for i in range(MIN_LEVEL, int(MAX_LEVEL / 2) + 1):
        for j in range(MIN_CLONE, int(MAX_CLONE / 2) + 1):
            lines[3] += f"{i},"
            lines[4] += f"{j},"
            lines[6] += f"{round(calcExperience(i, j, hatchLevel, chargeMultiplier, speciesMultiplier), 2)},"

    lines[3] = lines[3][:-1]
    lines[4] = lines[4][:-1]
    lines[6] = lines[6][:-1]

    finalContent = ""
    for i in range(len(lines)):
        finalContent += lines[i] + "\n"

    with open(file_name, 'w') as file:
        file.write(finalContent)

def genData(hatchLevel: HatchLevel, chargeMultiplier: ChargeMultiplier, speciesMultiplier: SpeciesMultiplier):
    file_name = f"{DATA_DIR}DMTNN-{hatchLevel.name}-{chargeMultiplier.name}-{speciesMultiplier.name}.txt"
    content = ""

    for i in range(MIN_LEVEL, MAX_LEVEL + 1):
        for j in range(MIN_CLONE, MAX_CLONE + 1):
            content += f"Level: {i} | Reinforcement: {j} | Experience: {round(calcExperience(i, j, hatchLevel, chargeMultiplier, speciesMultiplier), 2)}\n"

    with open(file_name, 'w') as file:
        file.write(content)

def genDataSet():
    for hatchLevel in HatchLevel:
        for chargeMultiplier in ChargeMultiplier:
            for speciesMultiplier in SpeciesMultiplier:
                genData(hatchLevel, chargeMultiplier, speciesMultiplier)

def readDataSet(hatchLevel: HatchLevel, chargeMultiplier: ChargeMultiplier, speciesMultiplier: SpeciesMultiplier):
    file_name = f"{DATA_DIR}DMTNN-{hatchLevel.name}-{chargeMultiplier.name}-{speciesMultiplier.name}.txt"
    file = open(file_name)
    allLines = file.readlines()
    
    levels, reinforcements, experiences = [],[],[]
    
    for i in range(len(allLines)):
        status = allLines[i].split("|")
        levels.append(int(status[0].split(":")[1]))
        reinforcements.append(int(status[1].split(":")[1]))
        experiences.append(float(status[2].split(":")[1]))

    dataSet = pd.DataFrame({
        'Level': levels,
        'Reinforcement': reinforcements,
        'Experience': experiences
    })

    return dataSet

genDataForNeuralEducator(HatchLevel.LEVEL5, ChargeMultiplier.REGULAR_CHARGE, SpeciesMultiplier.DIFFERENT_SPECIES)
# genDataSet()
# readDataSet(HatchLevel.LEVEL5, ChargeMultiplier.REGULAR_CHARGE, SpeciesMultiplier.DIFFERENT_SPECIES)