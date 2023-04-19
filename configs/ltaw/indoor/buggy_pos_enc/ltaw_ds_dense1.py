from src.config.default import _CN as cfg

cfg.LTAW.COARSE.TEMP_BUG_FIX = False
cfg.LTAW.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.LTAW.MATCH_COARSE.SPARSE_SPVS = False

cfg.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12, 17, 20, 23, 26, 29]
