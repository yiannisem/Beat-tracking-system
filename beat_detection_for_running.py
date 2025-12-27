import os
import numpy as np
import copy
from matplotlib import pyplot as plt
import librosa
from scipy.signal import medfilt as medfilt
from scipy.ndimage import maximum_filter1d as maxfilt
import sys
import IPython.display as ipd

def rms_odf(filename, windowTime, hopTime):
    """Computes a baseline onset detection function using RMS magnitude.
    Parameters:
        filename: path of the input audio file
        windowTime: window size for analysis (in seconds)
        hopTime: hop size (in seconds)
    Returns:
        A vector containing the root mean square magnitude for each frame of audio
    """
    snd, rate = librosa.load(filename, sr=None)
    # copmute hop size in samples
    hop = round(rate * hopTime)
    # round up to next power of 2
    wlen = int(2 ** np.ceil(np.log2(rate * windowTime)))
    # pad both sides with zeros
    snd = np.concatenate([np.zeros(wlen//2), snd, np.zeros(wlen//2)])
    # work out number of frames
    frameCount = int(np.floor((len(snd) - wlen) / hop + 1))
    odf = np.zeros(frameCount)
    # apply Hamming window and compute RMS magnitude
    window = np.hamming(wlen)
    for i in range(frameCount):
        start = i * hop
        frame = np.fft.fft(snd[start: start+wlen] * window)
        odf[i] = np.sqrt(np.mean(np.power(np.abs(frame), 2)))
    # normalise
    mx = max(odf)
    if mx > 0:
        odf /= mx
    return odf

def getOnsets(odf, hopTime, wd=3, thr=0.04):
    """Performs peak-picking, filtering and thresholding on an onset detection
    function to compute the times of each onset.
    Parameters:
        odf: a vector of onset detection function values, one for each frame
        hopTime: time in seconds between frames of the ODF
        wd: width of filter for median and maximum filtering
        thr: threshold (minimum difference between a peak value and the median
             of the ODF for a peak to be accepted as an onset)
    Returns:
        A vector of onset times in seconds
    """
    t = np.arange(len(odf)) * hopTime
    # median filter
    medFiltODF = odf - medfilt(odf, wd)
    # max filter
    maxFiltODF = maxfilt(medFiltODF, wd, mode='nearest', axis=0)
    # peak picking with threshold
    threshold = [max(i, thr) for i in maxFiltODF]
    peakIndices = np.nonzero(np.greater_equal(medFiltODF, threshold))
    peakTimes = peakIndices[0] * hopTime # Convert frame indices to time
    plt.figure(figsize=(14, 3))
    plt.xlabel('Time (s)')
    plt.ylabel('RMS ODF')
    plt.plot(t, medFiltODF, 'c')
    plt.plot(t, threshold, 'y')
    return peakTimes

# function to be used in clustering
def relationship_factor(interval_ratio):
    weight = (6 - interval_ratio) if 1 <= interval_ratio <= 4 else (1 if 5 <= interval_ratio <= 8 else 0)
    return weight


def cluster_iois(onsets, cluster_width):

    clusters = []
    # compute iois for each pair of onsets
    for onseti in range(len(onsets)):
        for onsetj in range(onseti + 1, len(onsets)):
            ioi = abs(onsets[onsetj] - onsets[onseti])
            ioi = float(ioi)

            # Find the cluster whose mean interval is closest to a particular ioi
            best_cluster = None
            best_diff = float('inf')
            for cluster in clusters:
                diff = abs(cluster["mean"] - ioi)
                if diff < best_diff:
                    best_diff = diff
                    best_cluster = cluster

            # check if a close cluster exists (one that has mean at most cluster_width away from ioi)
            if best_cluster is not None and best_diff < cluster_width:
                # add ioi to the cluster
                best_cluster["intervals"].append(ioi)
                best_cluster["mean"] = float(np.mean(best_cluster["intervals"]))
            else:
                # create a new cluster
                clusters.append({"intervals": [ioi], "mean": float(ioi)})

    # after all iois have been assigned to clusters, merge clusters that are too close
    i = 0
    while i < len(clusters):
        j = i + 1
        while j < len(clusters):
            if abs(clusters[i]["mean"] - clusters[j]["mean"]) < cluster_width:
                clusters[i]["intervals"].extend(clusters[j]["intervals"])
                clusters[i]["mean"] = float(np.mean(clusters[i]["intervals"]))
                clusters.pop(j)
            else:
                j += 1
        i += 1

    # initialise score for each cluster to 0
    for cluster in clusters:
        cluster["score"] = 0
    # compute score for each cluster, using the relationship factor defined separatelt as a function
    for cluster1 in clusters:
        for cluster2 in clusters:
            for n in range(1, 9):
                if abs(cluster1["mean"] - n * cluster2["mean"]) < cluster_width:
                    cluster1["score"] += relationship_factor(n) * len(cluster2["intervals"])
    return clusters

def tempo_filtering(clusters):
    i = 0
    while i < len(clusters):
        if clusters[i]["mean"] < 0.26 or clusters[i]["mean"] > 2.4:
            clusters.pop(i)
        else:
            i += 1
    return clusters


class Agent:
    def __init__(self, tempo, phase, history, score):
        
        # the following are defined as in Dixon 2001
        self.tempo = float(tempo)
        self.phase = float(phase)
        self.history = [float(t) for t in history]
        self.score = float(score)

    # cloning required to rake care of branching when this is needed
    def clone(self):
        return Agent(self.tempo, self.phase, copy.deepcopy(self.history), self.score)

# function to prune duplicates whenever these arise
def prune_duplicates(agents, tempo_thresh=0.01, phase_thresh=0.02):
    unique_agents = []
    for agent in agents:
        duplicate_found = False
        for u in unique_agents:
            # a duplicate is defined when it has very similar tempo and phase to another agent (ie. within specified thresholds)
            if abs(agent.tempo - u.tempo) < tempo_thresh and abs(agent.phase - u.phase) < phase_thresh:
                if agent.score > u.score:
                    unique_agents.remove(u)
                    unique_agents.append(agent)
                duplicate_found = True
                break
        if not duplicate_found:
            unique_agents.append(agent)
    return unique_agents

def beat_tracking(events, tempo_clusters, startup_period=5.0, correction_factor=0.5):
    """
    Beat tracking uses instances of the Agent class, returning a tuple (best_history, best_score), 
    which gives the list of beats of the highest scoring agent and its score.
    """
    agents = []
    """
    An agent is created for each tempo cluster and for each event near the start of the piece
    (ie. within the defined startup period)
    """
    for cluster in tempo_clusters:
        for tempo in cluster["intervals"]:
            for event in events:
                if event["time"] <= startup_period:
                    agents.append(Agent(float(tempo), float(event["time"]), [float(event["time"])], float(event["salience"])))
                else:
                    break

    for event in events:
        if event["time"] <= startup_period:
            continue

        new_agents = []
        for agent in agents:
            # predict next beat based on the current tempo and phase and add a beat not matched to any onset if the nearest onset is more than 1.15 times the tempo away from the predicted beat
            while event["time"] - agent.phase > agent.tempo * 1.15:
                interp_time = agent.phase + agent.tempo
                agent.history.append(float(interp_time))
                agent.phase = float(interp_time)

            # computes error between predicted beat and onset and defines tolerances for correction
            n = round((event["time"] - agent.phase) / agent.tempo)
            predicted_time = agent.phase + n * agent.tempo
            Tol_inner = 0.04
            Tol_pre = 0.25 * agent.tempo
            Tol_post = 0.15 * agent.tempo
            error = event["time"] - predicted_time

            # if the error is within the inner tolerance, the agent's phase and tempo is updated to align with the nearest onset
            if abs(error) <= Tol_inner:
                relativeError = abs(error) / Tol_inner
                agent.phase = float(event["time"])
                agent.tempo = float(agent.tempo + correction_factor * error)
                agent.history.append(float(event["time"]))
                # higher onset salience is rewarded with a higher score
                agent.score += float(event["salience"] * (1 - 0.5 * relativeError))

            # if the error is between inner and outer tolerances, the already existing agent adjusts its tempo and phase to align with the nearest onset
            # and a new agent is created which does not accept the nearest onset as a beat
            elif (-Tol_pre <= error < -Tol_inner) or (Tol_inner < error <= Tol_post):
                relativeError = abs(error) / (Tol_pre if error < 0 else Tol_post)
                alternative_agent = agent.clone()
                agent.phase = float(event["time"])
                agent.tempo = float(agent.tempo + correction_factor * error)
                agent.history.append(float(event["time"]))
                agent.score += float(event["salience"] * (1 - relativeError))
                new_agents.append(alternative_agent)
            new_agents.append(agent)
        # pruning function called for duplicates
        agents = prune_duplicates(new_agents, tempo_thresh=0.01, phase_thresh=0.02)
        max_agents = 10  # limit the number of agents, otherwise their number explodes
        if len(agents) > max_agents:
            agents = sorted(agents, key=lambda a: a.score, reverse=True)[:max_agents]
    # the agent with the highest score is identified and its history and score are returned
    best_agent = max(agents, key=lambda a: a.score)
    return best_agent.history, best_agent.score


def save_beats_to_file(beats, input_filename):
    """
    Saves detected beats in a 1D array format with each beat on a new line.
    The file is named as "detected_beats_for_<input_filename>.txt".
    """
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    
    output_filename = f"detected_beats_for_{base_name}.txt"

    beats = np.sort(np.array(beats, dtype=np.float64))  # Ensure 1D array format
    np.savetxt(output_filename, beats, fmt="%.6f") # Save file with 6 decimal places precision

    print(f"Beats saved to {output_filename} in correct format.")



def beatTracker(inputFile):
    windowSize = 0.03
    hopSize = 0.01
    filterWidth = 5
    threshold = 0.04

    odf = rms_odf(inputFile, windowSize, hopSize)
    estimatedOnsets = getOnsets(odf, hopSize, filterWidth, threshold)

    clusters = cluster_iois(estimatedOnsets, 0.2)
    filtered_clusters = tempo_filtering(clusters)

    events_with_salience = []
    for onset in estimatedOnsets:
        frame_idx = int(round(onset / hopSize))
        salience = odf[frame_idx]
        events_with_salience.append({"time": float(onset), "salience": float(salience)})

    beats, _ = beat_tracking(events_with_salience, filtered_clusters, startup_period=5.0, correction_factor=0.5)
    
    # Pass inputFile to ensure the correct output filename is generated
    save_beats_to_file(beats, inputFile)
    print("The detected beats are", beats)

    return beats


if __name__=="__main__":

    # change the path for testing as required
    exampleFile = r"C:/DriveSync/Queen_Mary/Modules/Music_Informatics/MI_coursework1/data1/BallroomData/Samba/Albums-AnaBelen_Veneo-02.wav"
    beatTracker(exampleFile)