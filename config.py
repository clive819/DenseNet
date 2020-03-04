class DenseNetConfig(object):
    # number of groups for group normalization
    numGroups = 8

    compressionRate = .5

    growthRate = 32

    # number of conv blocks
    numBlocks = [6, 12, 24, 16]
