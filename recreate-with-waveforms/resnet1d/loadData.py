"""Load one Experiment 1 recording, z-score it, and split it the way the original notebooks do."""


import numpy as np


trialsPerClass = 10
trainRepetitions = [0, 1, 2, 5, 6, 7]
testRepetitions = [3, 4, 8, 9]


def loadZscoredTrainTestSplit(taskFolder, articulationManner, subjectNumber = 1):
    """taskFolder is "Phoneme" or "Words"; articulationManner is "Voiced" or "Unvoiced".

    Returns trainTrials (classes * 6, 22, 7500), trainLabels, testTrials (classes * 4, 22, 7500), testLabels,
    ordered class by class like trainFeatures / testFeatures in allPhonemesSPDNet.ipynb.
    """
    trials = np.load("../Experiment1/" + taskFolder + "/" + articulationManner + "Subject" + str(subjectNumber) + ".npy")

    ### ZSCORE START 
    # Same z-scoring as allPhonemesSPDNet.ipynb / wordsSPDNet.ipynb: per trial and channel, over time, with + 1e-5.
    mean = np.mean(trials, axis = -1)  # Takes mean along the time axis. So each trial gets their own mean according to time per channel. Axis determines what you are doing the operation over.
    std = np.std(trials, axis = -1) 

    # mean and std are (trial, 22). Each cell being the mean/std
    # trials is (trial,22,7500). We need to reshape mean and std to (trial,22,1) to allow for calculations (that one mean/std value will be broadcasted over all 7500)

    trials = (trials - mean[..., np.newaxis])/(std[..., np.newaxis] + 1e-5)
    ### ZSCORE END

    ### ORGANIZING TRIALS 
    # Trials are stored in blocks of 10 per class, so axis 1 of this view is the repetition number.
    numberClasses = trials.shape[0] // trialsPerClass
    trialsByClass = trials.reshape(numberClasses, trialsPerClass, *trials.shape[1:]) # * unpacks into separate arguments. So trialsq gets reshaped to (38,10,22,7500)

    trainTrials = trialsByClass[:, trainRepetitions].reshape(-1, *trials.shape[1:]).astype(np.float32)
    testTrials = trialsByClass[:, testRepetitions].reshape(-1, *trials.shape[1:]).astype(np.float32)
    ### ORGANIZING TRIALS END

    ### LABELS S
    # Generating labels for the trials we pre1pare1d.
    trainLabels = np.repeat(np.arange(numberClasses), len(trainRepetitions)) # arange: Takes a number and creates 0 index ids.
    testLabels = np.repeat(np.arange(numberClasses), len(testRepetitions))
    return trainTrials, trainLabels, testTrials, testLabels
    ### LABELS E
