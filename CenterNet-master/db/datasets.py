from db.coco import MSCOCO 
from db.ui5k import UI5K
from db.rico import RICO
datasets = {
    "MSCOCO": MSCOCO,
    "UI5K": UI5K,
    "rico": RICO,
    "rico2k": RICO,
    "rico10k": RICO,
    "ricotext": RICO,
}
